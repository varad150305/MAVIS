'use strict';
/* Gesture mode, browser side.
 * The camera is THIS device's camera. Frames go to the hub over a WebSocket, through the original gesture controller
 * (MediaPipe + classifier + stabilizer + state machine), and the result comes back here and is drawn on THIS screen.
 * Nothing is shown on the computer that runs the hub. */

const $ = (s) => document.querySelector(s);
const EMOJI = { one_finger: '☝️', two_fingers: '✌️', three_fingers: '🤟', thumbs_up: '👍', open_palm: '🖐', UNKNOWN: '✋', NO_HAND: '✋' };
const NAMES = { one_finger: 'One finger', two_fingers: 'Two fingers', three_fingers: 'Three fingers', thumbs_up: 'Thumbs up', open_palm: 'Open palm' };
const OPEN = { ocr: '/mavis/', translation: '/drishti/?mode=translate', price_comparison: '/drishti/?mode=prices' };
const OPEN_NAME = { ocr: 'OCR mode', translation: 'Drishti · Translate', price_comparison: 'Drishti · Prices' };
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const SEND_W = 640, JPEG_Q = 0.7, MIN_GAP_MS = 40;

const S = {
  stream: null, facing: 'user', mirrored: true, ws: null, menu: null, last: null,
  inflight: false, lastSend: 0, running: false, classic: false, voice: true, autoOpen: false,
  canvas: document.createElement('canvas'), retry: 0, bannerTimer: null, countdown: null, wake: null, devices: 0,
};
const store = {
  get(k, d) { try { const v = localStorage.getItem('gesture.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('gesture.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};

// ------------------------------------------------------------------ speech + log
function say(text) {
  $('#say').textContent = text;
  if (!S.voice || !('speechSynthesis' in window)) return;
  try { speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; speechSynthesis.speak(u); } catch (e) { /* no voice */ }
}
function logLine(text) {
  const ol = $('#log');
  if (ol.querySelector('.empty')) ol.innerHTML = '';
  const li = document.createElement('li');
  const t = new Date();
  li.innerHTML = `<time>${t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>`;
  li.append(document.createTextNode(text));
  ol.prepend(li);
  while (ol.children.length > 8) ol.lastChild.remove();
}
function banner(text) {
  const b = $('#banner'); b.textContent = text; b.classList.add('show');
  clearTimeout(S.bannerTimer); S.bannerTimer = setTimeout(() => b.classList.remove('show'), 2200);
}
function label(feature) { const f = (S.menu?.selection || []).find((x) => x.feature === feature); return f ? f.label : feature; }

// ------------------------------------------------------------------ events from the controller
function onEvent(e) {
  const L = e.label || label(e.feature);
  switch (e.event) {
    case 'feature.selection_pending':
      logLine(`Selected ${L} — waiting for 👍`); say(`${L} selected. Thumbs up to confirm.`); break;
    case 'feature.start_requested':
      logLine(`Confirmed ${L}`); say(`${L} confirmed.`); if (S.autoOpen) startCountdown(e.feature); break;
    case 'feature.selection_cancelled':
      logLine(e.reason === 'timeout' ? `${label(e.feature)} timed out` : `Cancelled ${label(e.feature)}`);
      say(e.reason === 'timeout' ? 'Selection timed out.' : 'Selection cancelled.'); break;
    case 'feature.stop_requested':
      logLine(`Stopped ${label(e.feature)}`); cancelCountdown(); break;
    case 'navigation.return_to_selection':
      banner({ stopped: 'Stopped — back to feature selection', cancelled: 'Selection cancelled', timeout: 'Confirmation timed out' }[e.reason] || 'Back to feature selection');
      if (e.reason === 'stopped') say('Stopped. Choose a feature.');
      break;
    default: break;
  }
}

// ------------------------------------------------------------------ rendering
function renderPill(rec, stab) {
  const pill = $('#pill');
  pill.className = 'pill';
  if (!rec) { $('#pillEmo').textContent = '✋'; $('#pillName').textContent = 'No hand'; $('#pillSub').textContent = 'Show one hand to the camera'; return; }
  if (rec.no_hand) { $('#pillEmo').textContent = '✋'; $('#pillName').textContent = 'Hand not clear'; $('#pillSub').textContent = rec.reason || 'Keep the whole hand in view'; return; }
  if (!rec.is_gesture) { $('#pillEmo').textContent = '✋'; $('#pillName').textContent = 'Not a gesture'; $('#pillSub').textContent = rec.reason || ''; return; }
  pill.classList.add(rec.confirmed ? 'ok' : 'hold');
  $('#pillEmo').textContent = EMOJI[rec.gesture] || '✋';
  $('#pillName').textContent = NAMES[rec.gesture] || rec.gesture;
  const pct = Math.round((rec.confirmed ? 1 : stab.progress) * 100);
  $('#pillSub').textContent = `${rec.meaning || ''} · ${rec.confirmed ? 'recognised ✓' : 'hold… ' + pct + '%'}`;
}

function renderFingers(rec) {
  const box = $('#fingers');
  if (!rec || !rec.fingers || !rec.fingers.length) { box.innerHTML = ''; return; }
  box.innerHTML = rec.fingers.map((f) => `<div class="finger ${f.state === '?' ? 'Q' : f.state}">${f.name[0].toUpperCase() + f.name.slice(1, 5)}<b><i style="width:${Math.round(f.score * 100)}%"></i></b></div>`).join('');
}

function holdOf(gesture, stab) { return stab.candidate === gesture ? stab.progress : 0; }

function renderState(p) {
  const c = p.controller, stab = p.stability, m = S.menu, card = $('#stateCard');
  if (!m) return;
  card.className = 'card';
  if (c.state === 'SELECTING_FEATURE') {
    const key = 'sel';
    if (card.dataset.k !== key) {
      card.dataset.k = key;
      card.innerHTML = `<div class="kicker">Step 1 of 2</div><div class="big">Select a feature</div><div class="tiles">${m.selection.map((s) =>
        `<div class="tile" data-g="${s.gesture}"><span class="emo">${EMOJI[s.gesture] || '✋'}</span><div><b>${s.label}</b><small>${s.hint}</small></div><i class="prog"></i></div>`).join('')}</div>`;
    }
    card.querySelectorAll('.tile').forEach((t) => {
      const v = holdOf(t.dataset.g, stab);
      t.classList.toggle('holding', v > 0);
      t.querySelector('.prog').style.width = (v * 100) + '%';
    });
  } else if (c.state === 'AWAITING_CONFIRMATION') {
    card.classList.add('state-confirm');
    const key = 'confirm:' + c.pending_feature;
    if (card.dataset.k !== key) {
      card.dataset.k = key;
      card.innerHTML = `<div class="kicker">Step 2 of 2 · Confirm</div><div class="big">${c.pending_label}?</div>
        <div class="step go"><span class="emo">${EMOJI[m.confirm_gesture]}</span><div><b>Thumbs up</b> to start it</div><i class="bar-prog" data-g="${m.confirm_gesture}"></i></div>
        <div class="step back"><span class="emo">${EMOJI[m.back_gesture]}</span><div><b>Open palm</b> to cancel · other finger count to change</div><i class="bar-prog" data-g="${m.back_gesture}"></i></div>
        <div class="timer"><i id="timerBar"></i></div><div class="empty" id="timerTxt"></div>`;
    }
    card.querySelectorAll('.bar-prog').forEach((b) => { b.style.width = (holdOf(b.dataset.g, stab) * 100) + '%'; });
    const left = c.confirm_left ?? 0;
    $('#timerBar').style.width = Math.max(0, Math.min(100, (left / c.confirm_timeout) * 100)) + '%';
    $('#timerTxt').textContent = `Expires in ${Math.ceil(left)} s`;
  } else if (c.state === 'FEATURE_ACTIVE') {
    card.classList.add('state-active');
    const key = 'active:' + c.active_feature;
    if (card.dataset.k !== key) {
      card.dataset.k = key;
      const href = OPEN[c.active_feature];
      card.innerHTML = `<div class="kicker">Feature confirmed</div><div class="big">${c.active_label}</div>
        ${href ? `<a class="open" href="${href}">Open ${OPEN_NAME[c.active_feature]} →</a>` : ''}
        <div class="step back"><span class="emo">${EMOJI[m.back_gesture]}</span><div><b>Open palm</b> to stop and choose again</div><i class="bar-prog" data-g="${m.back_gesture}"></i></div>`;
    }
    card.querySelectorAll('.bar-prog').forEach((b) => { b.style.width = (holdOf(b.dataset.g, stab) * 100) + '%'; });
  }
  $('#notice').textContent = c.notice || '';
}

function drawHand(p) {
  const cv = $('#overlay'), ctx = cv.getContext('2d');
  const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!p.hand) return;
  // The server works on a mirrored (selfie) frame, like the original app. With the front camera this page also shows
  // the video mirrored, so the points line up as they are; with the back camera the video is shown as-is.
  const flip = p.mirrored_input !== S.mirrored;
  const pts = p.hand.points.map(([x, y]) => [(flip ? 1 - x : x) * w, y * h]);
  const rec = p.recognition, ok = rec && rec.confirmed, active = rec && rec.is_gesture;
  const color = ok ? '#4ADE80' : active ? '#FBBF24' : 'rgba(246,241,231,.85)';
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = 7;
  BONES.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke(); });
  ctx.strokeStyle = color; ctx.lineWidth = 3.5;
  BONES.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke(); });
  pts.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x, y, [4, 8, 12, 16, 20].includes(i) ? 6 : 4, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#000'; ctx.stroke(); });
  // hold ring near the wrist
  const prog = ok ? 1 : (p.stability.candidate && rec && rec.gesture === p.stability.candidate ? p.stability.progress : 0);
  if (prog > 0) {
    const [x, y] = pts[0];
    ctx.beginPath(); ctx.arc(x, y, 22, 0, Math.PI * 2); ctx.strokeStyle = 'rgba(0,0,0,.5)'; ctx.lineWidth = 7; ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 22, -Math.PI / 2, -Math.PI / 2 + prog * Math.PI * 2); ctx.strokeStyle = color; ctx.lineWidth = 5; ctx.stroke();
  }
}

function render(p) {
  S.last = p;
  renderPill(p.recognition, p.stability);
  renderFingers(p.recognition);
  renderState(p);
  if (!S.classic) drawHand(p);
  const r = p.recognition;
  $('#meta').innerHTML = `${p.fps.toFixed(0)} fps<span class="long">${r && r.orientation ? ' · ' + r.orientation : ''}</span>`;
  (p.events || []).forEach(onEvent);
}

// ------------------------------------------------------------------ auto-open countdown
function startCountdown(feature) {
  const href = OPEN[feature]; if (!href) return;
  cancelCountdown();
  let n = 3;
  $('#cdTitle').textContent = OPEN_NAME[feature]; $('#cdNum').textContent = n;
  $('#countdown').classList.add('show');
  S.countdown = setInterval(() => {
    n -= 1; $('#cdNum').textContent = n;
    if (n <= 0) { cancelCountdown(); stopCamera(); location.href = href; }
  }, 1000);
}
function cancelCountdown() { clearInterval(S.countdown); S.countdown = null; $('#countdown').classList.remove('show'); }

// ------------------------------------------------------------------ connection
function connect() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/hub/gesture/ws');
  ws.binaryType = 'blob';
  S.ws = ws; S.inflight = false;
  ws.onopen = () => { S.retry = 0; $('#liveDot').classList.add('live'); };
  ws.onmessage = (m) => {
    if (typeof m.data !== 'string') {                  // classic view: the original OpenCV window, as a picture
      const img = $('#classicImg'); const old = img.src; img.onload = fitFrame; img.src = URL.createObjectURL(m.data); if (old) URL.revokeObjectURL(old); return;
    }
    let p; try { p = JSON.parse(m.data); } catch (e) { return; }
    if (p.type === 'hello') { S.menu = p; ws.send(JSON.stringify({ cmd: 'classic', on: S.classic })); return; }
    if (p.type === 'error') { S.inflight = false; showError(p.message); return; }
    if (p.type === 'frame') { S.inflight = false; render(p); }
  };
  ws.onclose = () => {
    $('#liveDot').classList.remove('live'); S.inflight = false;
    if (S.running) { S.retry = Math.min(S.retry + 1, 6); setTimeout(() => { if (S.running) connect(); }, 400 * S.retry); }
  };
}

function pump() {
  if (!S.running) return;
  requestAnimationFrame(pump);
  const v = $('#video'), ws = S.ws;
  if (!ws || ws.readyState !== 1 || S.inflight || document.hidden || !v.videoWidth) return;
  const now = performance.now(); if (now - S.lastSend < MIN_GAP_MS) return;
  const scale = Math.min(1, SEND_W / Math.max(v.videoWidth, v.videoHeight));
  const c = S.canvas; c.width = Math.round(v.videoWidth * scale); c.height = Math.round(v.videoHeight * scale);
  c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);      // raw (un-mirrored) camera frame, like a webcam
  S.inflight = true; S.lastSend = now;
  c.toBlob((b) => { if (b && ws.readyState === 1) ws.send(b); else S.inflight = false; }, 'image/jpeg', JPEG_Q);
}

// ------------------------------------------------------------------ camera
function showError(msg) { $('#cover').hidden = false; $('#err').textContent = msg; $('#startBtn').disabled = false; }

async function startCamera() {
  $('#err').textContent = '';
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showError('This browser only allows the camera on a secure page. Open the https:// link shown on the computer (scan its QR code).');
    return;
  }
  $('#startBtn').disabled = true;
  try {
    stopTracks();
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: S.facing, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (e) {
    showError(e.name === 'NotAllowedError' ? 'Camera permission was refused. Allow the camera for this site in the browser settings, then try again.'
      : e.name === 'NotFoundError' ? 'No camera was found on this device.' : 'Could not start the camera: ' + e.message);
    return;
  }
  const v = $('#video');
  v.srcObject = S.stream;
  const settings = S.stream.getVideoTracks()[0].getSettings();
  S.mirrored = settings.facingMode ? settings.facingMode === 'user' : true;    // laptops report nothing: treat as a selfie camera
  v.classList.toggle('mirror', S.mirrored);
  await v.play().catch(() => {});
  fitFrame();
  $('#cover').hidden = true; $('#startBtn').disabled = false;
  try { S.devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput').length; } catch (e) { S.devices = 1; }
  $('#flipBtn').hidden = S.devices < 2;
  try { S.wake = await navigator.wakeLock?.request('screen'); } catch (e) { /* not supported */ }
  if (!S.running) { S.running = true; connect(); requestAnimationFrame(pump); }
  if (!S.menu) logLine('Camera on — hold up 1, 2 or 3 fingers');
}
function stopTracks() { if (S.stream) S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
function stopCamera() { S.running = false; stopTracks(); try { S.ws && S.ws.close(); } catch (e) { /* closed */ } }

// the picture keeps its own shape: the camera's, or in classic view the original window's (camera + side panel)
function fitFrame() {
  const v = $('#video'), img = $('#classicImg');
  const [w, h] = S.classic && img.naturalWidth ? [img.naturalWidth, img.naturalHeight] : [v.videoWidth || 4, v.videoHeight || 3];
  const st = $('#stage'), sw = st.clientWidth, sh = st.clientHeight;
  if (!sw || !sh) return;
  const k = Math.min(sw / w, sh / h);                      // largest box with exactly the picture's shape
  const f = $('#frame'); f.style.width = Math.floor(w * k) + 'px'; f.style.height = Math.floor(h * k) + 'px';
  if (S.last && !S.classic) drawHand(S.last);
}
window.addEventListener('resize', fitFrame);
$('#video').addEventListener('loadedmetadata', fitFrame);

// ------------------------------------------------------------------ controls
function toggle(btn, on) { btn.setAttribute('aria-pressed', on ? 'true' : 'false'); }
$('#startBtn').addEventListener('click', startCamera);
$('#flipBtn').addEventListener('click', () => { S.facing = S.facing === 'user' ? 'environment' : 'user'; startCamera(); });
$('#voiceBtn').addEventListener('click', () => { S.voice = !S.voice; toggle($('#voiceBtn'), S.voice); store.set('voice', S.voice); if (!S.voice) try { speechSynthesis.cancel(); } catch (e) { /* none */ } });
$('#classicBtn').addEventListener('click', () => {
  S.classic = !S.classic; toggle($('#classicBtn'), S.classic); $('#stage').classList.toggle('classic', S.classic);
  if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify({ cmd: 'classic', on: S.classic }));
  fitFrame();
});
$('#autoOpen').addEventListener('change', (e) => { S.autoOpen = e.target.checked; store.set('autoOpen', S.autoOpen); });
$('#cdCancel').addEventListener('click', cancelCountdown);
document.addEventListener('visibilitychange', async () => { if (!document.hidden && S.running && !S.wake) { try { S.wake = await navigator.wakeLock?.request('screen'); } catch (e) { /* none */ } } });
window.addEventListener('pagehide', stopCamera);

S.voice = store.get('voice', true); toggle($('#voiceBtn'), S.voice);
S.autoOpen = store.get('autoOpen', false); $('#autoOpen').checked = S.autoOpen;
if (!window.isSecureContext) showError('This page is not secure, so the browser will not allow the camera. Open the https:// link shown on the computer (scan its QR code).');
