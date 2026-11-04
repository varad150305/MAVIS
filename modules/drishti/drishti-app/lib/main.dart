import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'screens/home_screen.dart';
import 'screens/settings_screen.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  // Get available cameras
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Camera error: $e');
  }

  runApp(const DrishtiApp());
}

class DrishtiApp extends StatelessWidget {
  const DrishtiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Drishti दृष्टि',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0A0A0F),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF7C3AED),
          secondary: Color(0xFFA78BFA),
          surface: Color(0xFF141420),
        ),
        useMaterial3: true,
        fontFamily: 'system-ui',
      ),
      home: cameras.isNotEmpty
          ? HomeScreen(cameras: cameras)
          : const NoCameraScreen(),
      routes: {
        '/settings': (ctx) => const SettingsScreen(),
      },
    );
  }
}

class NoCameraScreen extends StatelessWidget {
  const NoCameraScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.no_photography, size: 64, color: Colors.white30),
            SizedBox(height: 16),
            Text('No camera found', style: TextStyle(color: Colors.white54)),
          ],
        ),
      ),
    );
  }
}
