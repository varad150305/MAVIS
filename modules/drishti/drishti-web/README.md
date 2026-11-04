# Drishti दृष्टि — AI Eyes for India

> **AI-powered accessibility tool for blind and visually impaired people in India.**
> Point your camera at anything — Drishti describes the scene, reads text, detects hazards, identifies food, compares prices, and responds to voice commands in Hindi, Tamil, Telugu, and 6 more Indian languages.

---

## Deploy in 2 minutes

### Option A — Vercel (recommended — zero setup for your users)
1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **New Project** → import your repo
3. Set **Root Directory** to `drishti-web` → click **Deploy**
4. In your new Vercel project → **Settings → Environment Variables** → add
   `GROQ_API_KEY` = your free key from [console.groq.com](https://console.groq.com), then **redeploy**
5. Done — every visitor gets a fully working app with **no account, no key, no setup**.
   This matters a lot for this app specifically: asking a blind or elderly visitor
   to create a developer account and paste an API key is not realistic. This one
   env var, set once by you, removes that barrier for everyone who ever uses the site.
   (Optional: add `SARVAM_API_KEY` the same way for natural Indian-language voices —
   without it, the app still works, just with the browser's built-in voice.)

Only Vercel gets this zero-setup benefit, because it's the only option here that
runs the small serverless proxy in `api/groq.js`. If you deploy via B, C, or D
below, visitors will still see the "paste your own key" screen — that's fine for
testing, but not ideal for the people this app is actually for.

### Option B — GitHub Pages (also free, visitors need their own key)
1. Push this repo to GitHub
2. Go to **Settings → Pages → Source → GitHub Actions**
3. The workflow at `.github/workflows/deploy.yml` runs automatically on every push
4. Your site is live at `https://YOUR_USERNAME.github.io/drishti/`

### Option C — Netlify (drag & drop, no account needed)
1. Go to [netlify.com/drop](https://app.netlify.com/drop)
2. Drag the `drishti-web` folder onto the page
3. Done — live URL in 30 seconds

### Option D — Local (no hosting needed)
```
npx serve .
```
Open the printed URL on any device on the same Wi-Fi.

---

## Setup after deploying

1. Open the live URL in **Chrome** or **Edge** (required for camera + voice)
2. Paste your **Groq API key** → click Save  
   → Get a free key at [console.groq.com](https://console.groq.com) (no credit card)
3. *(Optional)* Paste your **Sarvam API key** for Indian-language TTS  
   → Get a free key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai)
4. Click **Start Camera**
5. Say a voice command or tap the camera

---

## Features

### 10 AI Modes
| Mode | Voice command | What it does |
|------|---------------|--------------|
| 🔍 Describe | "describe" / "बताओ" | Warm, conversational scene description |
| 📄 Read Text | "read this" / "पढ़ो" | Transcribes all text — signs, books, receipts, screens |
| 💵 Currency | "money" / "पैसा" | Identifies Indian notes and coins, totals amount |
| 🎨 Color | "color" / "रंग" | Exact color names for clothing, objects, surfaces |
| 👥 People | "people" / "लोग" | Describes each person — age, clothing, expression |
| 🗺 Navigate | "navigate" / "रास्ता" | Obstacle detection + one clear action instruction |
| 📦 Objects | "objects" | Lists objects with Indian Rupee price estimates |
| 💰 Prices | "price" / "कीमत" | Amazon.in · Flipkart · Meesho price comparison |
| 🍽 Food | "food" / "खाना" | Identifies dish, ingredients, calories, or reads packaged food label |
| 🚨 Safety | "safety" / "सुरक्षित" | SAFE / CAUTION / DANGER rating with explanation |

### Blind-first Voice Control
- **Always-on voice engine** — say commands hands-free, no button needed
- **10 modes by voice** in English, Hindi, and all 8 Indian languages
- **Q&A mode** — ask anything about what the camera sees: *"Is there a staircase?"*, *"Kaun sa brand hai?"*
- **Automatic text detection** — if a book or sign is visible in Describe mode, Drishti says *"I can see text — say 'read it' to hear what's written"*

### Smart Live Mode
- Motion-triggered auto-scan (not a dumb timer)
- Checks for scene changes every 1.5 seconds on a tiny 80×60 pixel diff
- Only calls the AI when something actually moves — zero wasted API calls
- Minimum 8s between calls (free tier friendly)

### Guardian Mode (proactive hazard watch)
- Say **"guardian mode"** or **"watch out for me"** to turn it on — from then on
  Drishti silently checks the scene every few seconds in the background and
  only interrupts you with a spoken warning if it actually finds a hazard.
  This is the closest software equivalent of a car's forward-collision warning:
  you don't have to keep asking "is it safe?" — it tells you when it isn't.

### Street-crossing check
- Say **"is it safe to cross"** for a dedicated GO / WAIT verdict focused
  specifically on traffic signals, approaching vehicles, and crosswalks —
  the single highest-stakes decision a blind pedestrian makes. It's designed
  to fail safe: if the signal or road isn't clearly visible, it always says
  WAIT rather than guess.

### Clock-position spatial awareness
- Every mode (describe, navigate, safety, people, objects) now reports
  direction as a clock position combined with distance — "a chair at your
  2 o'clock, about 3 meters away" — instead of vague left/right, the same way
  co-pilot and collision-avoidance systems communicate position.

### Live translation
- Say **"translate this"** pointed at any sign, menu, or document — Drishti
  reads the original text, names the language, and translates it into
  whichever language is selected in the header.

### Wake-word gate (optional)
- Toggle **"Wake word"** in the header to require saying **"Hey Drishti"**
  before every command — useful in noisy places to stop ambient conversation
  from accidentally triggering the app.

### Emergency SOS
- Say **"help me"** or **"emergency"** — triggers instantly
- **Triple-tap** the camera (3 taps within 500ms)
- Or press the red **🆘 SOS** button
- Gets GPS location → sends SMS with Google Maps link → calls your contact
- Set emergency contact number in the header (saved to localStorage)

### WhatsApp Share
- Every description, extracted text, and Q&A answer has a **📲 WhatsApp** button
- Opens `wa.me/?text=` with the AI output — works on mobile and desktop

### Snap History
- Last 5 snaps saved to localStorage (survives page reload)
- **📋 History** button with badge count
- Each item: Speak again · Share on WA · Copy

### Indian Language Support
- **9 languages**: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, English (India)
- Voice commands work in all 9 languages — both native script and romanized
- TTS via Sarvam AI (Indian voices) with Web Speech API fallback

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Vision AI | Groq API — Llama 4 Scout 17B (free tier) |
| TTS | Sarvam AI `bulbul:v1` — Indian voices |
| TTS fallback | Web Speech API (built into Chrome) |
| STT | Web Speech API — `en-IN` + selected regional language |
| Camera | `getUserMedia` — works on mobile browsers |
| Storage | `localStorage` (no backend, no database) |
| Hosting | Static HTML — deploy anywhere |

---

## Cost

Everything runs on **free tiers**:

| Service | Free tier | Cost beyond free |
|---------|-----------|-----------------|
| Groq API | 14,400 requests/day | ~₹0.001/request |
| Sarvam TTS | 500 chars/request, generous free tier | ~₹0.002/request |
| Hosting | Free on Vercel / GitHub Pages / Netlify | ₹0 |
| **Total at 100 uses/day** | **₹0** | ~₹5–10/month |

---

## Reliability notes (read before relying on this for real users)

- **Model IDs drift over time.** Groq periodically retires model versions. If every
  mode suddenly stops working, the first thing to check is
  [console.groq.com/docs/models](https://console.groq.com/docs/models) — the model
  dropdown in the header must only contain IDs listed there under "vision". This
  exact issue silently broke 3 of 4 dropdown options as of mid-2026 and was fixed
  by removing the dead ones; it can happen again.
- **"Does not exist or you do not have access to it"** is a different problem from
  the above — it means the model itself is real and current, but *your specific
  Groq account* hasn't been granted access to it (some preview models are gated
  per-account, not just globally deprecated). The app now retries automatically
  with the other model in the dropdown when this happens, and switches to
  whichever one actually works for your key. If BOTH models in the dropdown give
  this error, log into console.groq.com → check whether your account needs to
  accept terms or request access for preview models before they'll work via the API.
- **Every failure is now spoken, not just shown as text** — camera errors, network
  errors, API errors, mic permission denials. A blind user can't see a red status
  dot, so silent failures are treated as bugs here, not edge cases.
- **Distance and "exact" figures are model estimates**, not sensor measurements —
  there's no depth camera or LiDAR involved. Good enough to navigate by, not
  precise enough for anything safety-critical beyond "walk carefully."
- **OSRM's free routing server is a shared public demo**, not a guarantee — if
  street navigation becomes a heavily used feature, budget for a paid routing
  provider (Mapbox, Google Directions) before depending on it.

## Roadmap

- [ ] Barcode / QR code scanning mode
- [ ] Flutter Android + iOS app (code in `drishti-app/`)
- [ ] Offline mode — on-device model for basic descriptions
- [x] Turn-by-turn walking navigation — say "navigate to [place]" / "directions to [place]" for real GPS turn-by-turn (free, via OpenStreetMap + OSRM, no key needed). Note: runs on OSRM's public demo routing server, which is fine for testing but not guaranteed uptime — worth swapping for a paid routing provider before heavy real-world use.
- [ ] WhatsApp bot interface (no app install needed)
- [ ] NGO / school dashboard for caretakers

---

Built with ❤️ for India's 8 million visually impaired people.  
**Drishti** (दृष्टि) means *vision* in Sanskrit.
