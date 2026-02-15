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
      ResolutionPreset.medium,
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
    // Inizializziamo le Mat come null per gestirle nel finally
    cv.Mat? mat;
    cv.Mat? finalBoard;
    cv.Mat? finalEnhanced;

    try {
      mat = cv.imread(capturedFile.path);
      if (mat == null || mat.isEmpty) return File(capturedFile.path);

      // 1. Pre-processing
      final gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY);
      final blurred = cv.gaussianBlur(gray, (5, 5), 0);
      final edged = cv.canny(blurred, 50, 150);

      // 2. Trova contorni
      final (contours, _) = cv.findContours(
        edged,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
      );

      cv.VecPoint? bestApprox;
      if (contours.length > 0) {
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
      }

      // 3. Warp o Fallback (assicuriamoci che finalBoard sia valida)
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
        finalBoard = mat.clone(); // Fallback se non trova la scacchiera
      }

      // 4. Miglioramento (Uso di convertTo in modo sicuro)
      // Creiamo prima la mat di destinazione per evitare che imwrite riceva un vuoto
      finalEnhanced = cv.Mat.empty();
      finalBoard.convertTo(finalEnhanced.type, alpha: 1.3, beta: 10);

      // Se per qualche motivo convertTo fallisce, usiamo finalBoard direttamente
      if (finalEnhanced.isEmpty) {
        finalEnhanced.dispose();
        finalEnhanced = finalBoard.clone();
      }

      // 5. Salvataggio
      final Directory dir = await getTemporaryDirectory();
      final String path =
          "${dir.path}/board_${DateTime.now().millisecondsSinceEpoch}.jpg";

      // Verifichiamo un'ultima volta prima di imwrite
      if (!finalEnhanced.isEmpty) {
        cv.imwrite(path, finalEnhanced);
      } else {
        throw Exception("Matrice ancora vuota prima del salvataggio");
      }

      return File(path);
    } catch (e) {
      debugPrint("ERRORE OPENCV: $e");
      return File(
        capturedFile.path,
      ); // Ritorna l'originale in caso di errore totale
    } finally {
      // Pulizia rigorosa per evitare leak di memoria
      mat?.dispose();
      finalBoard?.dispose();
      finalEnhanced?.dispose();
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
    final state = context.findAncestorStateOfType<_CameraPageState>()!;
    return GridView.builder(
      padding: const EdgeInsets.all(10),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 10,
        mainAxisSpacing: 10,
      ),
      itemCount: state.photos.length,
      itemBuilder: (ctx, i) => ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.file(
          state.photos[i],
          fit: BoxFit.cover,
          key: ValueKey(state.photos[i].path),
        ),
      ),
    );
  }
}
