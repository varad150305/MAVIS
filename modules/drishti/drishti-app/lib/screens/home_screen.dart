import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import '../services/groq_service.dart';
import '../services/sarvam_service.dart';
import '../services/sos_service.dart';
import 'settings_screen.dart';

// ─── Mode definitions ──────────────────────────────────────────────────────
const _modes = [
  {'id': 'describe', 'label': 'Describe', 'icon': Icons.visibility},
  {'id': 'read', 'label': 'Read Text', 'icon': Icons.article},
  {'id': 'currency', 'label': 'Currency', 'icon': Icons.currency_rupee},
  {'id': 'color', 'label': 'Color', 'icon': Icons.palette},
  {'id': 'people', 'label': 'People', 'icon': Icons.people},
  {'id': 'navigate', 'label': 'Navigate', 'icon': Icons.navigation},
  {'id': 'objects', 'label': 'Objects', 'icon': Icons.category},
  {'id': 'prices', 'label': 'Prices', 'icon': Icons.sell},
];

// ─── Nav keywords for proactive prompt ─────────────────────────────────────
final _navRegex = RegExp(
    r'\b(door|corridor|path|stairs|staircase|hallway|exit|road|crosswalk)\b',
    caseSensitive: false);

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const HomeScreen({super.key, required this.cameras});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  // Camera
  CameraController? _cam;
  bool _cameraReady = false;

  // Mode
  String _mode = 'describe';

  // AI + TTS
  final _groq = GroqService();
  final _sarvam = SarvamService();

  // State
  bool _loading = false;
  String _description = '';
  bool _showNavPrompt = false;

  // Q&A
  Uint8List? _lastFrame;
  final _qController = TextEditingController();
  bool _listening = false;
  final _speech = stt.SpeechToText();
  bool _speechReady = false;

  // SOS triple-tap
  int _sosTaps = 0;
  Timer? _sosTimer;

  // Auto-scan
  bool _autoScan = false;
  Timer? _autoTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
    _initSpeech();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cam == null || !(_cam!.value.isInitialized)) return;
    if (state == AppLifecycleState.inactive) {
      _cam?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      _showSnack('Camera permission denied');
      return;
    }

    final back = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );

    final ctrl = CameraController(back, ResolutionPreset.medium,
        enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);

    try {
      await ctrl.initialize();
      if (!mounted) return;
      setState(() {
        _cam = ctrl;
        _cameraReady = true;
      });
    } catch (e) {
      _showSnack('Camera init failed: $e');
    }
  }

  Future<void> _initSpeech() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) return;
    _speechReady = await _speech.initialize();
  }

  // ─── Capture & Describe ────────────────────────────────────────────────
  Future<void> _snap() async {
    if (_loading || !_cameraReady || _cam == null) return;
    setState(() {
      _loading = true;
      _showNavPrompt = false;
    });
    _sarvam.stop();

    try {
      final xfile = await _cam!.takePicture();
      final bytes = await xfile.readAsBytes();
      _lastFrame = bytes;

      final result = await _groq.describeImage(bytes, _mode);

      setState(() {
        _description = result;
        _loading = false;
        if (_mode == 'describe' && _navRegex.hasMatch(result)) {
          _showNavPrompt = true;
        }
      });

      await _sarvam.speak(result);
    } catch (e) {
      setState(() {
        _description = 'Error: $e';
        _loading = false;
      });
    }
  }

  // ─── Q&A ───────────────────────────────────────────────────────────────
  Future<void> _ask(String question) async {
    if (question.trim().isEmpty) return;
    if (_lastFrame == null) {
      _showSnack('Take a photo first');
      return;
    }
    setState(() => _loading = true);
    _sarvam.stop();

    final answer = await _groq.askQuestion(_lastFrame!, question);
    setState(() {
      _description = answer;
      _loading = false;
    });
    await _sarvam.speak(answer);
    _qController.clear();
  }

  // ─── Voice input ───────────────────────────────────────────────────────
  Future<void> _startListening() async {
    if (!_speechReady) {
      _showSnack('Microphone not available');
      return;
    }
    setState(() => _listening = true);
    await _speech.listen(
      onResult: (r) {
        if (r.finalResult) {
          setState(() => _listening = false);
          _ask(r.recognizedWords);
        }
      },
      listenFor: const Duration(seconds: 15),
      pauseFor: const Duration(seconds: 3),
      localeId: 'en_IN',
    );
  }

  // ─── Auto scan ─────────────────────────────────────────────────────────
  void _toggleAutoScan() {
    setState(() => _autoScan = !_autoScan);
    if (_autoScan) {
      _autoTimer = Timer.periodic(const Duration(seconds: 8), (_) => _snap());
    } else {
      _autoTimer?.cancel();
    }
  }

  // ─── SOS triple-tap ────────────────────────────────────────────────────
  void _onSosTap() {
    _sosTaps++;
    _sosTimer?.cancel();
    _sosTimer = Timer(const Duration(milliseconds: 800), () {
      if (_sosTaps >= 3) _triggerSOS();
      _sosTaps = 0;
    });
  }

  Future<void> _triggerSOS() async {
    HapticFeedback.heavyImpact();
    _showSnack('SOS triggered!');
    await SosService.triggerSOS();
  }

  // ─── Nav prompt ────────────────────────────────────────────────────────
  Future<void> _requestNav() async {
    setState(() => _showNavPrompt = false);
    await _ask('Give me detailed navigation guidance for what you see.');
  }

  // ─── Helpers ───────────────────────────────────────────────────────────
  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg), duration: const Duration(seconds: 2)));
  }

  void _copyDescription() {
    if (_description.isEmpty) return;
    Clipboard.setData(ClipboardData(text: _description));
    _showSnack('Copied to clipboard');
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cam?.dispose();
    _sarvam.dispose();
    _qController.dispose();
    _autoTimer?.cancel();
    _sosTimer?.cancel();
    super.dispose();
  }

  // ─── Build ─────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0F),
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(),
            _buildCamera(),
            _buildModePills(),
            _buildDescriptionCard(),
            if (_showNavPrompt) _buildNavPrompt(),
            _buildBottomBar(),
          ],
        ),
      ),
    );
  }

  // ─── Top bar ───────────────────────────────────────────────────────────
  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        children: [
          // SOS button — triple tap
          GestureDetector(
            onTap: _onSosTap,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFDC2626),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Text('SOS',
                  style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 13)),
            ),
          ),
          const Spacer(),
          // App title
          const Text('दृष्टि',
              style: TextStyle(
                  color: Color(0xFFA78BFA),
                  fontSize: 22,
                  fontWeight: FontWeight.bold)),
          const Spacer(),
          // Settings
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white54),
            onPressed: () =>
                Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const SettingsScreen())),
          ),
        ],
      ),
    );
  }

  // ─── Camera preview ────────────────────────────────────────────────────
  Widget _buildCamera() {
    if (!_cameraReady || _cam == null) {
      return Expanded(
        child: Container(
          color: Colors.black,
          child: const Center(
            child: CircularProgressIndicator(color: Color(0xFF7C3AED)),
          ),
        ),
      );
    }

    return Expanded(
      child: ClipRRect(
        borderRadius: BorderRadius.circular(0),
        child: CameraPreview(_cam!),
      ),
    );
  }

  // ─── Mode pills ────────────────────────────────────────────────────────
  Widget _buildModePills() {
    return SizedBox(
      height: 48,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        itemCount: _modes.length,
        itemBuilder: (ctx, i) {
          final m = _modes[i];
          final selected = _mode == m['id'];
          return Padding(
            padding: const EdgeInsets.only(right: 6),
            child: GestureDetector(
              onTap: () {
                setState(() {
                  _mode = m['id'] as String;
                  _showNavPrompt = false;
                });
                HapticFeedback.selectionClick();
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: selected
                      ? const Color(0xFF7C3AED)
                      : const Color(0xFF1E1E2E),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: selected
                          ? const Color(0xFF7C3AED)
                          : Colors.white12),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(m['icon'] as IconData,
                        size: 14,
                        color: selected ? Colors.white : Colors.white54),
                    const SizedBox(width: 4),
                    Text(m['label'] as String,
                        style: TextStyle(
                            color: selected ? Colors.white : Colors.white54,
                            fontSize: 12,
                            fontWeight: selected
                                ? FontWeight.w600
                                : FontWeight.normal)),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ─── Description card ──────────────────────────────────────────────────
  Widget _buildDescriptionCard() {
    return Container(
      constraints: const BoxConstraints(maxHeight: 140),
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF141420),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: _loading
          ? const Center(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Color(0xFF7C3AED))),
                  SizedBox(width: 10),
                  Text('Analysing…',
                      style:
                          TextStyle(color: Colors.white54, fontSize: 13)),
                ],
              ),
            )
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Expanded(
                      child: Text('Description',
                          style: TextStyle(
                              color: Colors.white38,
                              fontSize: 11,
                              fontWeight: FontWeight.w500)),
                    ),
                    if (_description.isNotEmpty) ...[
                      // Re-read button
                      GestureDetector(
                        onTap: () => _sarvam.speak(_description),
                        child: const Icon(Icons.volume_up,
                            size: 18, color: Color(0xFFA78BFA)),
                      ),
                      const SizedBox(width: 10),
                      // Copy button
                      GestureDetector(
                        onTap: _copyDescription,
                        child: const Icon(Icons.copy,
                            size: 16, color: Colors.white38),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 6),
                Expanded(
                  child: SingleChildScrollView(
                    child: Semantics(
                      label: _description.isEmpty
                          ? 'No description yet. Tap Snap to analyse the scene.'
                          : _description,
                      child: Text(
                        _description.isEmpty
                            ? 'Tap Snap to describe what the camera sees.'
                            : _description,
                        style: TextStyle(
                            color: _description.isEmpty
                                ? Colors.white24
                                : Colors.white,
                            fontSize: 13,
                            height: 1.5),
                      ),
                    ),
                  ),
                ),
              ],
            ),
    );
  }

  // ─── Nav prompt ────────────────────────────────────────────────────────
  Widget _buildNavPrompt() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A30),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF7C3AED)),
      ),
      child: Row(
        children: [
          const Icon(Icons.navigation, color: Color(0xFF7C3AED), size: 18),
          const SizedBox(width: 8),
          const Expanded(
            child: Text('Navigation help available',
                style: TextStyle(color: Colors.white70, fontSize: 13)),
          ),
          GestureDetector(
            onTap: _requestNav,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: const Color(0xFF7C3AED),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Text('Guide me',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w600)),
            ),
          ),
        ],
      ),
    );
  }

  // ─── Bottom bar ────────────────────────────────────────────────────────
  Widget _buildBottomBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      child: Column(
        children: [
          // Q&A row
          Row(
            children: [
              Expanded(
                child: Container(
                  height: 40,
                  decoration: BoxDecoration(
                    color: const Color(0xFF1E1E2E),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: TextField(
                    controller: _qController,
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                    decoration: const InputDecoration(
                      hintText: 'Ask a question…',
                      hintStyle: TextStyle(color: Colors.white30),
                      border: InputBorder.none,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    ),
                    onSubmitted: _ask,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // Mic button
              GestureDetector(
                onTap: _listening ? null : _startListening,
                child: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: _listening
                        ? const Color(0xFFDC2626)
                        : const Color(0xFF1E1E2E),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white12),
                  ),
                  child: Icon(
                    _listening ? Icons.mic : Icons.mic_none,
                    color: Colors.white,
                    size: 18,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          // Action row — Auto scan + Snap
          Row(
            children: [
              // Auto scan toggle
              GestureDetector(
                onTap: _toggleAutoScan,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    color: _autoScan
                        ? const Color(0xFF4C1D95)
                        : const Color(0xFF1E1E2E),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                        color: _autoScan
                            ? const Color(0xFF7C3AED)
                            : Colors.white12),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _autoScan
                            ? Icons.pause_circle_outline
                            : Icons.play_circle_outline,
                        color: Colors.white,
                        size: 16,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        _autoScan ? 'Scanning' : 'Auto',
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.w500),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // Snap button (big, centre)
              Expanded(
                child: GestureDetector(
                  onTap: _snap,
                  child: Semantics(
                    label: 'Snap and describe',
                    button: true,
                    child: Container(
                      height: 52,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF7C3AED), Color(0xFF4F46E5)],
                        ),
                        borderRadius: BorderRadius.circular(26),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF7C3AED).withOpacity(0.4),
                            blurRadius: 12,
                            offset: const Offset(0, 4),
                          )
                        ],
                      ),
                      child: _loading
                          ? const Center(
                              child: SizedBox(
                                width: 22,
                                height: 22,
                                child: CircularProgressIndicator(
                                    strokeWidth: 2.5,
                                    color: Colors.white),
                              ),
                            )
                          : const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.camera_alt,
                                    color: Colors.white, size: 20),
                                SizedBox(width: 8),
                                Text('Snap',
                                    style: TextStyle(
                                        color: Colors.white,
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold)),
                              ],
                            ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
