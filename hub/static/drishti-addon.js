'use strict';
/* Added by MAVIS on top of the Drishti page (Drishti's own index.html is unchanged):
 *   - a "Modes" button back to the launcher;
 *   - /drishti/?mode=prices (or translate, text, ...) opens Drishti straight in that mode. Gesture mode uses this:
 *     two fingers + thumbs up -> Translate, three fingers + thumbs up -> Prices;
 *   - /drishti/?prepare=1 (used by the "Prepare for offline" page): loads each of Drishti's on-device models once with
 *     Drishti's own loader functions, so that this computer's offline mirror saves them for use without internet. */
(function () {
  if (new URLSearchParams(location.search).get('prepare') === '1') { prepare(); return; }
  const home = document.createElement('a');
  home.className = 'ds-home'; home.href = '/';
  home.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>Modes';
  const settings = document.querySelector('header .hbtn[onclick*="showSettingsModal"]');
  if (settings) { home.classList.add('ds-inline'); settings.parentNode.insertBefore(home, settings); } else document.body.appendChild(home);

  const mode = new URLSearchParams(location.search).get('mode');
  if (!mode || !/^[a-z]+$/.test(mode) || !document.getElementById('pill-' + mode)) return;
  const NAMES = { prices: 'Prices', translate: 'Translate', text: 'Read Text', describe: 'Describe', currency: 'Currency' };
  function go() {
    try { if (typeof launchApp === 'function' && document.getElementById('landing')) launchApp(); } catch (e) { /* already open */ }
    try { setMode(mode); } catch (e) { return; }
    const t = document.createElement('div'); t.className = 'ds-toast'; t.setAttribute('role', 'status');
    const camOn = document.getElementById('startBtn') && document.getElementById('startBtn').disabled;
    t.innerHTML = `Opened in <b>${NAMES[mode] || mode}</b> mode.` + (camOn ? '' : ' Tap <b>Start Camera</b> to begin.');
    document.body.appendChild(t); setTimeout(() => t.remove(), 5000);
    if (new URLSearchParams(location.search).get('autostart') === '1') autostart();
  }
  // opened by gesture control: start the camera and describe the scene once, without any tap
  async function autostart() {
    try {
      const sb = document.getElementById('startBtn');
      if (sb && !sb.disabled && typeof startCam === 'function') await startCam();
      for (let i = 0; i < 40; i++) {                      // wait for the first camera frames
        const v = document.querySelector('video');
        if (v && v.videoWidth) break;
        await new Promise((r) => setTimeout(r, 250));
      }
      await new Promise((r) => setTimeout(r, 800));
      if (typeof snap === 'function') snap();
    } catch (e) { /* the user can still tap the picture */ }
  }
  if (document.readyState === 'complete') setTimeout(go, 300); else window.addEventListener('load', () => setTimeout(go, 300));

  async function prepare() {
    const say = (step, state, detail) => { try { parent.postMessage({ type: 'mavis-prepare', step, state, detail: detail || '' }, location.origin); } catch (e) { /* no parent */ } };
    // Drishti's own loaders and ready flags (top-level declarations in its page script)
    const steps = [
      ['loadObjectDetector', () => loadObjectDetector(), () => odReady],
      ['loadOcrWorker', () => loadOcrWorker(), () => ocrReady],
      ['loadSoundClassifier', () => loadSoundClassifier(), () => soundReady],
      ['loadDepthModel', () => loadDepthModel(), () => depthReady],
      ['loadLocalVLM', () => loadLocalVLM(), () => localVlmReady],
    ];
    await new Promise((r) => (document.readyState === 'complete' ? r() : window.addEventListener('load', r)));
    for (const [name, run, ok] of steps) {
      say(name, 'loading');
      try { await run(); } catch (e) { /* the flag below tells */ }
      let ready = false; try { ready = !!ok(); } catch (e) { ready = false; }
      say(name, ready ? 'ready' : 'failed');
    }
    say('all', 'done');
  }
})();

/* ---- MAVIS: Drishti without its OCR and translation modes ----------------------------------------------
   In MAVIS, documents are read by Varad's OCR + LLM module and speech is translated by Yogita's live translation
   module. Drishti keeps everything else. Its Read Text and Translate modes are hidden here (Drishti's files are not
   changed), and any route into them (pill, voice command, ?mode=) points to the MAVIS module instead. */
(function () {
  'use strict';
  const MOVED = {
    text: { label: 'Read Text', to: 'OCR + LLM', url: '/mavis/' },
    translate: { label: 'Translate', to: 'Live Translation', url: '/translate/' },
  };
  const css = document.createElement('style');
  css.textContent = '#pill-text, #pill-translate { display: none !important; }';
  document.head.appendChild(css);
  function notice(m) {
    const info = MOVED[m];
    let el = document.getElementById('mavis-moved');
    if (!el) {
      el = document.createElement('div'); el.id = 'mavis-moved';
      el.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:99999;background:#1b1f2a;color:#fff;padding:12px 18px;border-radius:12px;font:600 15px system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.3);max-width:92vw;text-align:center';
      document.body.appendChild(el);
    }
    const inControl = window.top !== window;
    el.innerHTML = `${info.label} is handled by <b>${info.to}</b> in MAVIS. ` +
      (inControl ? 'Show 🖐 to turn Drishti off, then 👍 and the other gesture.' : `<a href="${info.url}" style="color:#5eead4">Open it</a>`);
    clearTimeout(el._t); el._t = setTimeout(() => el.remove(), 6000);
    try { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(`${info.label} is handled by ${info.to.replace(/\(.*\)/, '')} in MAVIS.`)); } catch (e) { /* no voice */ }
  }
  function wrap(name) {
    const orig = window[name];
    if (typeof orig !== 'function') return;
    window[name] = function (m, ...rest) {
      if (MOVED[m]) { notice(m); return; }
      return orig.call(this, m, ...rest);
    };
  }
  function install() { wrap('setMode'); wrap('setModeVoice'); }
  if (document.readyState === 'complete') install(); else window.addEventListener('load', install);
  install();
})();
