import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SarvamService {
  static const _endpoint = 'https://api.sarvam.ai/text-to-speech';

  // Speaker and model fixed for quality
  static const _speaker = 'meera';
  static const _model = 'bulbul:v1';

  final AudioPlayer _player = AudioPlayer();
  final FlutterTts _tts = FlutterTts();
  bool _ttsReady = false;

  SarvamService() {
    _initTts();
  }

  Future<void> _initTts() async {
    await _tts.setLanguage('hi-IN');
    await _tts.setSpeechRate(0.9);
    await _tts.setVolume(1.0);
    _ttsReady = true;
  }

  Future<String?> _getSarvamKey() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('drishti_sarvam_key');
  }

  Future<String> _getLang() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('drishti_lang') ?? 'hi-IN';
  }

  Future<void> speak(String text) async {
    if (text.isEmpty) return;

    final key = await _getSarvamKey();
    final lang = await _getLang();

    if (key != null && key.isNotEmpty) {
      final success = await _speakViaSarvam(text, key, lang);
      if (success) return;
    }

    // Fallback to flutter_tts
    await _speakViaTts(text, lang);
  }

  Future<bool> _speakViaSarvam(
      String text, String key, String lang) async {
    try {
      // Sarvam max 500 chars per request
      final chunk = text.length > 500 ? text.substring(0, 500) : text;

      final res = await http
          .post(
            Uri.parse(_endpoint),
            headers: {
              'Content-Type': 'application/json',
              'api-subscription-key': key,
            },
            body: jsonEncode({
              'inputs': [chunk],
              'target_language_code': lang,
              'speaker': _speaker,
              'model': _model,
              'enable_preprocessing': true,
            }),
          )
          .timeout(const Duration(seconds: 15));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final audio = data['audios']?[0] as String?;
        if (audio != null) {
          final bytes = base64Decode(audio);
          await _playAudioBytes(bytes);
          return true;
        }
      }
    } catch (_) {}
    return false;
  }

  Future<void> _playAudioBytes(Uint8List bytes) async {
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/drishti_tts.wav');
    await file.writeAsBytes(bytes);
    await _player.stop();
    await _player.play(DeviceFileSource(file.path));
  }

  Future<void> _speakViaTts(String text, String lang) async {
    if (!_ttsReady) return;
    // Map Sarvam lang codes to BCP-47
    final bcp47 = _toBcp47(lang);
    await _tts.setLanguage(bcp47);
    await _tts.stop();
    await _tts.speak(text);
  }

  String _toBcp47(String sarvamCode) {
    const map = {
      'hi-IN': 'hi-IN',
      'ta-IN': 'ta-IN',
      'te-IN': 'te-IN',
      'kn-IN': 'kn-IN',
      'ml-IN': 'ml-IN',
      'bn-IN': 'bn-IN',
      'mr-IN': 'mr-IN',
      'gu-IN': 'gu-IN',
      'en-IN': 'en-IN',
    };
    return map[sarvamCode] ?? 'hi-IN';
  }

  Future<void> stop() async {
    await _player.stop();
    await _tts.stop();
  }

  void dispose() {
    _player.dispose();
    _tts.stop();
  }
}
