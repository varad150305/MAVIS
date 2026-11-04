# Drishti Flutter App — Setup Guide (Windows)

## Step 1: Install Flutter

1. Go to **https://docs.flutter.dev/get-started/install/windows**
2. Click **Download Flutter SDK** → extract the zip to `C:\flutter`
3. Add `C:\flutter\bin` to your PATH:
   - Search → "Environment Variables" → Edit **Path** → Add `C:\flutter\bin`
4. Open a new **PowerShell** window and run:
   ```
   flutter doctor
   ```
   You'll see a checklist. Fix any ❌ items (instructions in the output).

---

## Step 2: Install Android Studio (for Android)

1. Download from **https://developer.android.com/studio**
2. Install with default settings — this installs the Android SDK automatically
3. Open Android Studio → **More Actions → SDK Manager**
4. Install **Android SDK 34** (or latest)
5. Run `flutter doctor` again — the Android ✓ should be green

---

## Step 3: Accept Android licenses

```powershell
flutter doctor --android-licenses
# Press 'y' to accept each one
```

---

## Step 4: Connect your Android phone

1. On your phone: **Settings → About Phone → tap Build Number 7 times** (enables Developer Mode)
2. **Settings → Developer Options → USB Debugging → ON**
3. Plug in via USB
4. Run: `flutter devices` — your phone should appear

---

## Step 5: Run Drishti on your phone

```powershell
cd C:\Drishti\drishti-app
flutter pub get
flutter run
```

The app will build and install on your connected phone (~2-3 minutes first time).

---

## Step 6: Add your API key in the app

1. Open the app → tap **Settings** (gear icon, top-right)
2. Paste your **Groq API key** (get free at https://console.groq.com)
3. Optionally add **Sarvam API key** for Indian-language voice (https://dashboard.sarvam.ai)
4. Choose your preferred language
5. Add emergency contacts for SOS
6. Tap **Save Settings**

---

## iOS Setup (if you have a Mac)

1. Install Xcode from the Mac App Store
2. Run: `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`
3. Connect your iPhone → trust the Mac on your iPhone
4. In the `drishti-app` folder: `flutter run -d <your-iphone-device-id>`
5. First run requires signing in Xcode: open `ios/Runner.xcworkspace` → set your Apple ID in Signing & Capabilities

> **Note:** iOS builds require a Mac. You can distribute the Android APK directly without a Mac.

---

## Build Release APK (to share with others)

```powershell
cd C:\Drishti\drishti-app
flutter build apk --release
```

Output: `build\app\outputs\flutter-apk\app-release.apk`
Share this file — anyone can install it on Android.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `flutter: command not found` | Restart terminal after adding to PATH |
| `No devices found` | Enable USB debugging, try a different USB cable |
| Camera black screen | Grant camera permission in phone Settings → Apps → Drishti |
| Voice not working | Grant microphone permission in phone Settings → Apps → Drishti |
| Build fails: SDK not found | Run `flutter doctor` and follow Android SDK instructions |

---

## Features in the App

- **8 modes**: Describe, Read Text, Currency, Color, People, Navigate, Objects, Prices
- **Voice Q&A**: Tap mic or type to ask follow-up questions about the scene
- **Auto Scan**: Continuously describes the scene every 8 seconds
- **Navigation prompt**: Appears automatically when a door/path/stairs is detected
- **SOS**: Triple-tap the red SOS button to SMS + call emergency contacts
- **Indian languages**: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, English
- **TalkBack compatible**: All interactive elements have accessibility labels
