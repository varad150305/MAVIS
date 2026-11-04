import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/groq_service.dart';
import '../services/sos_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _groqCtrl = TextEditingController();
  final _sarvamCtrl = TextEditingController();
  final _sosCtrl = TextEditingController();

  String _model = GroqService.defaultModel;
  String _lang = 'hi-IN';
  bool _saving = false;

  static const _languages = [
    {'code': 'hi-IN', 'name': 'हिन्दी (Hindi)'},
    {'code': 'ta-IN', 'name': 'தமிழ் (Tamil)'},
    {'code': 'te-IN', 'name': 'తెలుగు (Telugu)'},
    {'code': 'kn-IN', 'name': 'ಕನ್ನಡ (Kannada)'},
    {'code': 'ml-IN', 'name': 'മലയാളം (Malayalam)'},
    {'code': 'bn-IN', 'name': 'বাংলা (Bengali)'},
    {'code': 'mr-IN', 'name': 'मराठी (Marathi)'},
    {'code': 'gu-IN', 'name': 'ગુજરાતી (Gujarati)'},
    {'code': 'en-IN', 'name': 'English (India)'},
  ];

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _groqCtrl.text = prefs.getString('drishti_groq_key') ?? '';
      _sarvamCtrl.text = prefs.getString('drishti_sarvam_key') ?? '';
      _model = prefs.getString('drishti_model') ?? GroqService.defaultModel;
      _lang = prefs.getString('drishti_lang') ?? 'hi-IN';
    });
    final contacts = await SosService.loadContacts();
    _sosCtrl.text = contacts.join(', ');
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    final prefs = await SharedPreferences.getInstance();

    await prefs.setString('drishti_groq_key', _groqCtrl.text.trim());
    await prefs.setString('drishti_sarvam_key', _sarvamCtrl.text.trim());
    await prefs.setString('drishti_model', _model);
    await prefs.setString('drishti_lang', _lang);

    // Parse SOS contacts
    final nums = _sosCtrl.text
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    await SosService.saveContacts(nums);

    setState(() => _saving = false);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Settings saved ✓')));
    Navigator.pop(context);
  }

  @override
  void dispose() {
    _groqCtrl.dispose();
    _sarvamCtrl.dispose();
    _sosCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0F),
      appBar: AppBar(
        backgroundColor: const Color(0xFF141420),
        title: const Text('Settings',
            style: TextStyle(color: Colors.white, fontSize: 18)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white70),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _section('API Keys'),
          _field(
            controller: _groqCtrl,
            label: 'Groq API Key',
            hint: 'gsk_…',
            helper: 'Get free key at console.groq.com',
            obscure: true,
          ),
          const SizedBox(height: 12),
          _field(
            controller: _sarvamCtrl,
            label: 'Sarvam API Key (optional)',
            hint: 'For Indian-language voice',
            helper: 'Get free key at dashboard.sarvam.ai',
            obscure: true,
          ),
          const SizedBox(height: 20),

          _section('AI Model'),
          _dropdown<String>(
            value: _model,
            items: GroqService.models
                .map((m) => DropdownMenuItem(
                      value: m,
                      child: Text(m.split('/').last,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 13)),
                    ))
                .toList(),
            onChanged: (v) => setState(() => _model = v!),
          ),
          const SizedBox(height: 20),

          _section('Voice Language'),
          _dropdown<String>(
            value: _lang,
            items: _languages
                .map((l) => DropdownMenuItem(
                      value: l['code']!,
                      child: Text(l['name']!,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 13)),
                    ))
                .toList(),
            onChanged: (v) => setState(() => _lang = v!),
          ),
          const SizedBox(height: 20),

          _section('Emergency SOS Contacts'),
          _field(
            controller: _sosCtrl,
            label: 'Phone numbers',
            hint: '+91xxxxxxxxxx, +91xxxxxxxxxx',
            helper: 'Comma-separated. Triple-tap SOS button to trigger.',
            obscure: false,
          ),
          const SizedBox(height: 32),

          // Save button
          SizedBox(
            height: 52,
            child: ElevatedButton(
              onPressed: _saving ? null : _save,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF7C3AED),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(26)),
              ),
              child: _saving
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child:
                          CircularProgressIndicator(strokeWidth: 2.5, color: Colors.white))
                  : const Text('Save Settings',
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold)),
            ),
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _section(String title) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Text(title,
            style: const TextStyle(
                color: Color(0xFFA78BFA),
                fontSize: 12,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.8)),
      );

  Widget _field({
    required TextEditingController controller,
    required String label,
    required String hint,
    required String helper,
    required bool obscure,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      style: const TextStyle(color: Colors.white, fontSize: 13),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: Colors.white54, fontSize: 13),
        hintText: hint,
        hintStyle: const TextStyle(color: Colors.white24, fontSize: 12),
        helperText: helper,
        helperStyle: const TextStyle(color: Colors.white38, fontSize: 11),
        filled: true,
        fillColor: const Color(0xFF1E1E2E),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Color(0xFF7C3AED)),
        ),
      ),
    );
  }

  Widget _dropdown<T>({
    required T value,
    required List<DropdownMenuItem<T>> items,
    required ValueChanged<T?> onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E2E),
        borderRadius: BorderRadius.circular(10),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<T>(
          value: value,
          items: items,
          onChanged: onChanged,
          dropdownColor: const Color(0xFF1E1E2E),
          isExpanded: true,
        ),
      ),
    );
  }
}
