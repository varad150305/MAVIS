'use strict';
/* Added by MAVIS on top of the MAVIS page (MAVIS's own files are unchanged):
 *   - a "Modes" button back to the launcher;
 *   - "Use this device's camera": this phone/laptop camera becomes MAVIS's camera. Frames go to the hub, which serves
 *     them to MAVIS in DroidCam's format, so MAVIS scans exactly as it does with DroidCam. */
(function () {
  const home = document.createElement('a');
  const ARROW = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  // Narrow screens (sidebar hidden): a small button at the top right. Wide screens: a link at the top of MAVIS's own
  // sidebar, so it never covers MAVIS's buttons (e.g. "Stop scanning" sits at the top right of the page).
  home.className = 'ds-home ds-top ds-narrow-only'; home.href = '/';
  home.innerHTML = ARROW + 'Modes';
  document.body.appendChild(home);
  function sideLink() {
    const sb = document.getElementById('sidebar');
    if (!sb || sb.querySelector('.ds-side-home')) return;
    const a = document.createElement('a');
    a.className = 'ds-side-home'; a.href = '/'; a.innerHTML = ARROW + 'All modes · MAVIS';
    const brand = sb.querySelector('.brand');
    if (brand) brand.after(a); else sb.prepend(a);
  }
  new MutationObserver(sideLink).observe(document.getElementById('sidebar') || document.body, { childList: true, subtree: true });
  sideLink();

  // 1080p for small print (as MAVIS recommends for DroidCam); 15 fps, since MAVIS calls anything under 12 fps "slow".
  const W = 1920, H = 1080, FPS = 15, Q = 0.85;
  const cam = { stream: null, ws: null, timer: null, watch: null, video: null, canvas: document.createElement('canvas'), badge: null, busy: false };
  // The camera's <video> lives on <body>, not inside MAVIS's form: MAVIS replaces the whole screen when the scan
  // starts, and a video element removed from the page stops delivering new frames (the picture would freeze).
  function cameraVideo() {
    if (!cam.video) {
      cam.video = document.createElement('video');
      cam.video.playsInline = true; cam.video.muted = true; cam.video.setAttribute('aria-hidden', 'true');
      cam.video.style.cssText = 'position:fixed;left:0;bottom:0;width:2px;height:2px;opacity:0.01;pointer-events:none;z-index:-1';
      document.body.appendChild(cam.video);
    }
    return cam.video;
  }

  function stop() {
    clearInterval(cam.timer); clearInterval(cam.watch); cam.timer = cam.watch = null;
    if (cam.stream) cam.stream.getTracks().forEach((t) => t.stop());
    cam.stream = null; cam.busy = false;
    if (cam.video) cam.video.srcObject = null;
    try { cam.ws && cam.ws.close(); } catch (e) { /* closed */ }
    cam.ws = null;
    if (cam.badge) { cam.badge.remove(); cam.badge = null; }
  }

  function sendFrame() {
    const v = cam.video, ws = cam.ws;
    // skip this tick while the previous frame is still being encoded or sent: the rate adapts to the phone and Wi-Fi
    if (cam.busy || !v || !v.videoWidth || !ws || ws.readyState !== 1 || ws.bufferedAmount > 1e6) return;
    cam.busy = true;
    const c = cam.canvas; c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob((b) => { cam.busy = false; if (b && ws.readyState === 1) ws.send(b); }, 'image/jpeg', Q);
  }

  async function start(box, phone = true) {
    const err = (m) => { box.querySelector('.ds-err').textContent = m || ''; };
    const btn = box.querySelector('button');
    err('');
    box.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      err('The browser only allows the camera on a secure page. Open the https:// link shown on the computer.'); return;
    }
    btn.disabled = true;
    stop();
    try {
      const video = { facingMode: { ideal: 'environment' }, width: { ideal: W }, height: { ideal: H } };
      const id = window.MAVIS_findCamera ? await window.MAVIS_findCamera(phone) : null;
      if (id) video.deviceId = { exact: id };
      cam.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      err(e.name === 'NotAllowedError' ? 'Camera permission was refused.' : 'Could not start the camera: ' + e.message);
      box.querySelectorAll('button').forEach((b) => { b.disabled = false; }); return;
    }
    const v = cameraVideo(); v.srcObject = cam.stream; await v.play().catch(() => {});
    cam.ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/hub/camera/ws');
    await new Promise((res, rej) => { cam.ws.onopen = res; cam.ws.onerror = rej; }).catch(() => {});
    if (!cam.ws || cam.ws.readyState !== 1) { err('Could not connect to the computer.'); stop(); box.querySelectorAll('button').forEach((b) => { b.disabled = false; }); return; }
    cam.timer = setInterval(sendFrame, 1000 / FPS);
    cam.badge = document.createElement('div'); cam.badge.className = 'ds-live';
    const label = (cam.stream.getVideoTracks()[0] || {}).label || 'camera';
    cam.badge.innerHTML = '<i></i>' + (/droidcam/i.test(label) ? 'Phone camera (DroidCam)' : 'Camera: ' + label.replace(/\(.*\)/, '').trim());
    document.body.appendChild(cam.badge);
    await new Promise((r) => setTimeout(r, 1200));            // a few frames first, so MAVIS can open the stream at once
    const sel = document.querySelector('#selector'), iv = document.querySelector('#interval');
    const body = { selector: sel ? sel.value : 'b0', interval_s: iv ? Number(iv.value) || 5 : 5 };
    try {
      const r = await fetch('/hub/mavis/phone-scan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error?.message || 'MAVIS could not start the scan.'); }
    } catch (e) { err(e.message); stop(); box.querySelectorAll('button').forEach((b) => { b.disabled = false; }); return; }
    // keep sending while MAVIS is scanning; stop by itself when the scan ends (MAVIS's own Stop / Cancel buttons)
    let seenScan = false;
    cam.watch = setInterval(async () => {
      try {
        const s = await (await fetch('/mavis/api/state', { cache: 'no-store' })).json();
        const ph = s.scan && s.scan.phase;
        const scanning = ph === 'connecting' || ph === 'scanning';
        if (scanning) seenScan = true;
        else if (seenScan || !s.scan) stop();
      } catch (e) { /* keep trying */ }
    }, 1000);
  }

  function inject() {
    const form = document.querySelector('#camForm');
    if (!form || form.parentElement.querySelector('.ds-cam')) return;
    const box = document.createElement('div');
    box.className = 'ds-cam';
    box.innerHTML = '<h3>📱 Use this device\'s camera</h3><p>Scan with the camera of this laptop (or of the phone, if MAVIS is open on the phone). To stream from the DroidCam phone app over Wi-Fi instead, use the form below.</p>'
      + '<button type="button" class="btn primary lg block">Start scanning with this camera</button><div class="ds-err" role="alert"></div>';
    box.querySelector('button').addEventListener('click', () => start(box, false));
    form.parentElement.insertBefore(box, form);
  }
  new MutationObserver(inject).observe(document.getElementById('main') || document.body, { childList: true, subtree: true });
  inject();
  window.addEventListener('pagehide', stop);
})();
