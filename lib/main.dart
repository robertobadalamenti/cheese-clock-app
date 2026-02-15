import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    theme: ThemeData.dark(),
    home: const CameraPage(),
    debugShowCheckedModeBanner: false,
  );
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});
  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? controller;
  bool isProcessing = false;
  List<File> photos = [];

  @override
  void initState() {
    super.initState();
    initializeCamera();
  }

  Future<void> initializeCamera() async {
    if (_cameras.isEmpty) return;
    controller = CameraController(
      _cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
    );
    await controller!.initialize();
    if (mounted) setState(() {});
  }

  // Ordina i punti per il Warp
  cv.VecPoint _sortPoints(cv.VecPoint src) {
    final points = List.generate(src.length, (i) => src[i]);
    points.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = points.first;
    final br = points.last;
    points.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    final tr = points.first;
    final bl = points.last;
    return cv.VecPoint.fromList([tl, tr, br, bl]);
  }

  Future<File> processWithOpenCV(XFile capturedFile) async {
    cv.Mat? mat;
    cv.Mat? finalBoard;
    cv.Mat? finalEnhanced;

    try {
      mat = cv.imread(capturedFile.path);
      if (mat.isEmpty) return File(capturedFile.path);

      final gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY);

      // --- LIVELLO 1: Rilevamento Griglia ---
      final patternSize = (7, 7);
      final (foundGrid, corners) = cv.findChessboardCorners(
        gray,
        patternSize,
        flags: cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
      );

      if (foundGrid && corners.length == 49) {
        debugPrint("🎯 Griglia 8x8 rilevata!");
        final srcPoints = cv.VecPoint.fromList([
          cv.Point(corners[0].x.toInt(), corners[0].y.toInt()),
          cv.Point(corners[6].x.toInt(), corners[6].y.toInt()),
          cv.Point(corners[48].x.toInt(), corners[48].y.toInt()),
          cv.Point(corners[42].x.toInt(), corners[42].y.toInt()),
        ]);

        final dstPoints = cv.VecPoint.fromList([
          cv.Point(0, 0),
          cv.Point(800, 0),
          cv.Point(800, 800),
          cv.Point(0, 800),
        ]);

        final M = cv.getPerspectiveTransform(srcPoints, dstPoints);
        finalBoard = cv.warpPerspective(mat, M, (800, 800));

        M.dispose();
        srcPoints.dispose();
        dstPoints.dispose();
      } else {
        // --- LIVELLO 2: Fallback Contorni ---
        debugPrint("⚠️ Griglia non trovata. Provo con i contorni...");
        final blurred = cv.gaussianBlur(gray, (5, 5), 0);
        final edged = cv.canny(blurred, 50, 150);
        final (contours, _) = cv.findContours(
          edged,
          cv.RETR_EXTERNAL,
          cv.CHAIN_APPROX_SIMPLE,
        );

        cv.VecPoint? bestApprox;
        double maxArea = 0;

        for (int i = 0; i < contours.length; i++) {
          final area = cv.contourArea(contours[i]);
          if (area > 2000) {
            final peri = cv.arcLength(contours[i], true);
            final approx = cv.approxPolyDP(contours[i], 0.02 * peri, true);
            if (approx.length == 4 && area > maxArea) {
              bestApprox?.dispose();
              bestApprox = _sortPoints(approx);
              maxArea = area;
            } else {
              approx.dispose();
            }
          }
        }

        if (bestApprox != null) {
          final dstPoints = cv.VecPoint.fromList([
            cv.Point(0, 0),
            cv.Point(800, 0),
            cv.Point(800, 800),
            cv.Point(0, 800),
          ]);
          final M = cv.getPerspectiveTransform(bestApprox!, dstPoints);
          finalBoard = cv.warpPerspective(mat, M, (800, 800));
          M.dispose();
          dstPoints.dispose();
          bestApprox!.dispose();
        } else {
          finalBoard = mat.clone();
        }
        edged.dispose();
        blurred.dispose();
        contours.dispose();
      }

      // --- CORREZIONE CONVERT TO ---
      // Usiamo la firma: Mat convertTo(MatType type, {double alpha, double beta})
      finalEnhanced = finalBoard.convertTo(
        finalBoard.type,
        alpha: 1.2,
        beta: 5,
      );

      // Controllo di sicurezza: se finalEnhanced è venuta vuota, usa finalBoard
      if (finalEnhanced.isEmpty) {
        debugPrint("⚠️ Errore nel miglioramento, uso l'immagine grezza.");
        await saveMatToGrid(finalBoard);
      } else {
        await saveMatToGrid(finalEnhanced);
      }

      gray.dispose();
      return File(capturedFile.path);
    } catch (e) {
      debugPrint("ERRORE CRITICO: $e");
      return File(capturedFile.path);
    } finally {
      mat?.dispose();
      finalBoard?.dispose();
      finalEnhanced?.dispose();
    }
  }

  Future<void> saveMatToGrid(cv.Mat imageMat) async {
    if (imageMat.isEmpty) {
      debugPrint("Errore: Tentativo di salvare una Mat vuota");
      return;
    }

    try {
      // 1. Ottieni la directory temporanea
      final Directory dir = await getTemporaryDirectory();

      // 2. Crea un nome file unico usando il timestamp per evitare conflitti di cache
      final String path =
          "${dir.path}/processed_${DateTime.now().millisecondsSinceEpoch}.jpg";

      // 3. Scrivi l'immagine sul disco (imwrite restituisce true se riesce)
      final success = cv.imwrite(path, imageMat);

      if (success) {
        // 4. Aggiorna la UI inserendo il nuovo File nella lista photos
        setState(() {
          photos.insert(0, File(path));
        });
        debugPrint("Immagine salvata e aggiunta alla griglia: $path");
      } else {
        debugPrint("Errore durante l'esecuzione di imwrite");
      }
    } catch (e) {
      debugPrint("Errore nel salvataggio del file: $e");
    }
  }

  Future<void> captureAndProcess() async {
    if (controller == null || isProcessing) return;

    setState(() => isProcessing = true);

    try {
      final XFile file = await controller!.takePicture();
      final File processed = await processWithOpenCV(file);

      setState(() {
        photos.insert(0, processed);
        isProcessing = false;
      });
    } catch (e) {
      debugPrint("Errore: $e");
      setState(() => isProcessing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: const Text("Chess Scanner")),
      body: Column(
        children: [
          AspectRatio(
            aspectRatio: controller!.value.aspectRatio,
            child: CameraPreview(controller!),
          ),
          const SizedBox(height: 20),
          isProcessing
              ? const CircularProgressIndicator()
              : ElevatedButton.icon(
                  onPressed: captureAndProcess,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("SCATTA E ELABORA"),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 30,
                      vertical: 15,
                    ),
                  ),
                ),
          Expanded(
            child: _PhotosGrid(), // Widget separato per pulizia
          ),
        ],
      ),
    );
  }
}

class _PhotosGrid extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // Recuperiamo lo stato per accedere alla lista photos
    final state = context.findAncestorStateOfType<_CameraPageState>()!;

    return GridView.builder(
      padding: const EdgeInsets.all(10),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 10,
        mainAxisSpacing: 10,
      ),
      itemCount: state.photos.length,
      itemBuilder: (ctx, i) => GestureDetector(
        // Al tocco apriamo la visualizzazione modale
        onTap: () => _showImagePreview(context, state.photos[i]),
        child: Hero(
          tag: state.photos[i].path, // Tag per animazione fluida (opzionale)
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.file(
              state.photos[i],
              fit: BoxFit.cover,
              key: ValueKey(state.photos[i].path),
            ),
          ),
        ),
      ),
    );
  }

  // Funzione per mostrare il dialogo modale
  void _showImagePreview(BuildContext context, File imageFile) {
    showDialog(
      context: context,
      barrierDismissible: true, // Chiudi se tocchi fuori
      builder: (ctx) => Dialog(
        backgroundColor: Colors.black,
        insetPadding: const EdgeInsets.all(10),
        child: Stack(
          alignment: Alignment.topRight,
          children: [
            // InteractiveViewer permette lo zoom e il pan dell'immagine
            InteractiveViewer(
              panEnabled: true,
              minScale: 0.5,
              maxScale: 5.0,
              child: Container(
                width: double.infinity,
                height: double.infinity,
                child: Image.file(imageFile, fit: BoxFit.contain),
              ),
            ),
            // Tasto per chiudere la modale
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: CircleAvatar(
                backgroundColor: Colors.black54,
                child: IconButton(
                  icon: const Icon(Icons.close, color: Colors.white),
                  onPressed: () => Navigator.of(ctx).pop(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
