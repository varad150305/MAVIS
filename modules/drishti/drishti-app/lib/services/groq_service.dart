import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class GroqService {
  static const _endpoint =
      'https://api.groq.com/openai/v1/chat/completions';

  static const _models = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'meta-llama/llama-3.2-11b-vision-preview',
    'meta-llama/llama-3.2-90b-vision-preview',
  ];

  static String get defaultModel => _models[0];
  static List<String> get models => _models;

  // Prompts — mirrors the web app exactly
  static const _prompts = {
    'describe':
        'You are Drishti, an AI assistant for blind people. Describe this image clearly. '
            'Do all of the following: '
            '(1) In 1-2 sentences describe the main subject and scene. '
            '(2) Transcribe ALL visible text exactly — documents, signs, handwriting, screens, labels. '
            '(3) If identifiable products are visible, give their estimated prices in Indian Rupees on Amazon.in and Flipkart. '
            '(4) If you see a door, path, stairs, or corridor, describe it and ask "Would you like navigation guidance?" '
            'Write in plain sentences only. No bullet points, no headers.',
    'read':
        'You are Drishti. Read and transcribe ALL text visible in this image exactly as written — '
            'signs, documents, menus, labels, screens, handwriting. '
            'Preserve line breaks. No commentary, just the text.',
    'currency':
        'You are Drishti. Identify every currency note or coin visible. '
            'State the denomination and total amount. '
            'Write in plain sentences.',
    'color':
        'You are Drishti. Describe the dominant colors of the main object or scene. '
            'Be specific — fabric color, object color, background color. '
            'Plain sentences only.',
    'people':
        'You are Drishti. Count and describe every person visible. '
            'For each person: gender, approximate age, notable features '
            '(glasses, beard, head covering, clothing color, if using a phone). '
            'Plain sentences only.',
    'navigate':
        'You are Drishti helping a blind person navigate. '
            'Identify: nearest obstacles, doors, stairs, paths, crossings. '
            'Give clear instructions: how far, which direction, any hazards. '
            'Plain sentences only.',
    'objects':
        'You are Drishti. List all objects visible. '
            'For each: what it is, rough price in Indian Rupees if it is a product. '
            'Plain sentences only.',
    'prices':
        'You are Drishti. Identify any products visible. '
            'Give estimated price in Indian Rupees on Amazon.in and Flipkart. '
            'Plain sentences only.',
  };

  final List<Map<String, dynamic>> _chatHistory = [];
  String _lastDescription = '';

  String get lastDescription => _lastDescription;

  void clearHistory() {
    _chatHistory.clear();
    _lastDescription = '';
  }

  Future<String?> _getKey() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('drishti_groq_key');
  }

  Future<String?> _getModel() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('drishti_model') ?? defaultModel;
  }

  /// Describe the scene in the given mode.
  Future<String> describeImage(Uint8List imageBytes, String mode) async {
    final key = await _getKey();
    if (key == null || key.isEmpty) {
      return 'No Groq API key. Please add it in Settings.';
    }
    final model = await _getModel() ?? defaultModel;

    final prompt = _prompts[mode] ?? _prompts['describe']!;
    final b64 = base64Encode(imageBytes);

    final body = jsonEncode({
      'model': model,
      'temperature': 0.2,
      'max_tokens': 700,
      'messages': [
        {
          'role': 'user',
          'content': [
            {'type': 'text', 'text': prompt},
            {
              'type': 'image_url',
              'image_url': {'url': 'data:image/jpeg;base64,$b64'},
            }
          ]
        }
      ]
    });

    final result = await _post(key, body);

    if (mode == 'describe' || mode == 'navigate') {
      _lastDescription = result;
    }
    return result;
  }

  /// Follow-up question with chat history.
  Future<String> askQuestion(
      Uint8List imageBytes, String question) async {
    final key = await _getKey();
    if (key == null || key.isEmpty) {
      return 'No Groq API key. Please add it in Settings.';
    }
    final model = await _getModel() ?? defaultModel;

    // Build message list
    final messages = <Map<String, dynamic>>[
      {
        'role': 'system',
        'content': 'You are Drishti, an AI assistant for blind people. '
            'Answer questions about images clearly and concisely.'
            '${_lastDescription.isNotEmpty ? ' Last scene: $_lastDescription' : ''}',
      }
    ];

    // Add recent history (last 6 messages = 3 exchanges)
    messages.addAll(_chatHistory.length > 6
        ? _chatHistory.sublist(_chatHistory.length - 6)
        : _chatHistory);

    final b64 = base64Encode(imageBytes);
    messages.add({
      'role': 'user',
      'content': [
        {'type': 'text', 'text': question},
        {
          'type': 'image_url',
          'image_url': {'url': 'data:image/jpeg;base64,$b64'},
        }
      ]
    });

    final body = jsonEncode({
      'model': model,
      'temperature': 0.2,
      'max_tokens': 500,
      'messages': messages,
    });

    final answer = await _post(key, body);

    // Update history
    _chatHistory.add({
      'role': 'user',
      'content': question,
    });
    _chatHistory.add({
      'role': 'assistant',
      'content': answer,
    });

    return answer;
  }

  Future<String> _post(String key, String body) async {
    try {
      final res = await http
          .post(
            Uri.parse(_endpoint),
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $key',
            },
            body: body,
          )
          .timeout(const Duration(seconds: 30));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return data['choices']?[0]?['message']?['content']?.toString().trim() ??
            'No response.';
      } else if (res.statusCode == 429) {
        return 'Rate limit hit. Please wait a moment and try again.';
      } else if (res.statusCode == 401) {
        return 'Invalid Groq API key. Please check Settings.';
      } else {
        final err = jsonDecode(res.body);
        return 'Error: ${err['error']?['message'] ?? res.statusCode}';
      }
    } catch (e) {
      return 'Network error: $e';
    }
  }
}
