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
