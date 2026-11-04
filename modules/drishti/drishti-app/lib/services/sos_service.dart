import 'package:geolocator/geolocator.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

class SosService {
  static const _contactKey = 'drishti_sos_contacts';

  /// Save emergency contact numbers (comma-separated)
  static Future<void> saveContacts(List<String> numbers) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_contactKey, numbers.join(','));
  }

  /// Load emergency contact numbers
  static Future<List<String>> loadContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_contactKey) ?? '';
    if (raw.isEmpty) return [];
    return raw.split(',').where((s) => s.trim().isNotEmpty).toList();
  }

  /// Trigger SOS — get location then SMS + call first contact
  static Future<void> triggerSOS() async {
    final contacts = await loadContacts();
    if (contacts.isEmpty) return;

    String locationText = 'Location unavailable';
    try {
      final perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.always ||
          perm == LocationPermission.whileInUse) {
        final pos = await Geolocator.getCurrentPosition(
          desiredAccuracy: LocationAccuracy.high,
        ).timeout(const Duration(seconds: 10));
        locationText =
            'https://maps.google.com/?q=${pos.latitude},${pos.longitude}';
      }
    } catch (_) {}

    final first = contacts.first.trim();
    final msg = Uri.encodeComponent(
        'EMERGENCY: I need help! My location: $locationText — sent via Drishti app');

    // SMS first contact
    final smsUri = Uri.parse('sms:$first?body=$msg');
    if (await canLaunchUrl(smsUri)) {
      await launchUrl(smsUri);
    }

    // Slight delay then call
    await Future.delayed(const Duration(seconds: 2));
    final callUri = Uri.parse('tel:$first');
    if (await canLaunchUrl(callUri)) {
      await launchUrl(callUri);
    }
  }
}
