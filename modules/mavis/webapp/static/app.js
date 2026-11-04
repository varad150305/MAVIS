'use strict';
/* MAVIS web UI: no build step, no external requests. State comes from /api/events (server-sent events). */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const ico = (name, cls = '') => `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const S = {
  app: null, docs: [], docsVersion: -1, skew: 0, route: { name: 'home' }, doc: null, docLoading: null,
  screen: null, view: null, ui: { selPage: null, tab: 'ocr', docOpen: false, sidebarOpen: false, pending: null },
  handoff: {}, lastCaptureAt: null, notice: null,
};
const serverNow = () => Date.now() / 1000 + S.skew;
const fmtDur = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return m ? `${m}m ${String(Math.floor(s % 60)).padStart(2, '0')}s` : `${s.toFixed(1)}s`; };
const fmtClock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });
const fmtDate = (iso) => { const d = new Date(iso); return isNaN(d) ? '' : d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ', ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); };
const pct = (v) => (v == null ? '-' : Math.round(v * 100) + '%');

async function api(url, opts = {}) {
  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  let data = null;
  try { data = await res.json(); } catch (e) { /* not json */ }
  if (!res.ok) { const err = new Error((data && data.error && data.error.message) || `Request failed (${res.status})`); err.code = data && data.error && data.error.code; throw err; }
  return data;
}
function toast(message, bad = false) {
  const el = document.createElement('div'); el.className = 'toast' + (bad ? ' bad' : ''); el.textContent = message;
  $('#toasts').appendChild(el); setTimeout(() => el.remove(), 5200);
}
function setText(root, sel, value) { const el = $(sel, root); if (el && el.textContent !== String(value)) el.textContent = value; }
function setHTML(root, sel, value) { const el = $(sel, root); if (el && el.innerHTML !== value) el.innerHTML = value; }

/* ------------------------------------------------------------------ events */
function connectEvents() {
  const es = new EventSource('/api/events');
  es.onmessage = (ev) => { S.app = JSON.parse(ev.data); S.skew = S.app.now - Date.now() / 1000; $('#offline').hidden = true; onState(); };
  es.onerror = () => { $('#offline').hidden = false; };
}

async function refreshDocs() {
  try { S.docs = (await api('/api/documents')).documents; renderSidebar(); } catch (e) { /* shown via offline banner */ }
}

function onState() {
  const app = S.app;
  if (app.docs_version !== S.docsVersion) { S.docsVersion = app.docs_version; refreshDocs(); }
  const screen = pickScreen();
  if (screen !== S.screen) { S.screen = screen; mountScreen(screen); renderSidebar(); }
  else if (S.view && S.view.update) S.view.update();
  renderSidebarStatus();
}

function pickScreen() {
  const sc = S.app.scan;
  if (!sc) S.homeOverride = false;
  if (sc) {
    // The user pressed the MAVIS logo: show Home (or a document) whatever the scan is doing. The scan carries on in the
    // background and Home offers a banner to go back to it.
    if (S.homeOverride && (sc.phase === 'connecting' || sc.phase === 'scanning' || sc.phase === 'finalizing' ||
        (sc.phase === 'finished' && !S.handoff[sc.id]))) return routeScreen();
    if (sc.phase === 'connecting' || sc.phase === 'scanning') return 'scan';
    if (sc.phase === 'finalizing') return 'final';
    if (sc.phase === 'finished') {
      if (S.handoff[sc.id]) return routeScreen();       // the user (or the auto-open below) already left this screen
      if (sc.empty) return 'final';
      const prep = S.app.prepare[sc.id];
      if (prep && prep.status === 'ready' && !S.openTimer) {
        S.openTimer = setTimeout(() => { S.openTimer = null; enterDocument(sc.id); }, 1100);   // let "ready" be seen
      }
      return 'final';
    }
    if ((sc.phase === 'failed' || sc.phase === 'cancelled') && !S.handoff[sc.id]) {
      S.handoff[sc.id] = true;
      if (sc.phase === 'failed') S.notice = sc.error;
      api('/api/scan/dismiss', { method: 'POST' }).catch(() => {});
    }
  }
  return routeScreen();
}
function routeScreen() { return S.route.name === 'doc' ? 'doc' : 'home'; }

function goHome() {
  S.homeOverride = true; S.ui.sidebarOpen = false;
  if (location.hash === '#/' || location.hash === '' || location.hash === '#') onRoute(); else location.hash = '#/';
}
function backToScan() {
  S.homeOverride = false;
  const sc = S.app && S.app.scan; const prep = sc && S.app.prepare[sc.id];
  if (sc && sc.phase === 'finished' && !sc.empty && prep && prep.status === 'ready') { enterDocument(sc.id); return; }
  S.route = { name: 'home' };
  if (location.hash !== '#/') history.replaceState(null, '', '#/');
  S.screen = pickScreen(); mountScreen(S.screen); renderSidebar();
}

function enterDocument(id) {
  if (S.handoff[id]) return;
  S.handoff[id] = true;
  S.homeOverride = false;
  S.doc = null;
  api('/api/scan/dismiss', { method: 'POST' }).catch(() => {});
  if (location.hash === '#/doc/' + id) onRoute(); else location.hash = '#/doc/' + id;
}

function mountScreen(screen) {
  const main = $('#main'); main.innerHTML = ''; S.view = null;
  const views = { home: HomeView, scan: ScanView, final: FinalView, doc: DocView };
  S.view = views[screen](main);
  if (S.view.update) S.view.update();
  document.title = { home: 'MAVIS - AI Document Assistant', scan: 'Scanning - MAVIS', final: 'Finalizing - MAVIS', doc: (S.doc ? S.doc.title + ' - ' : '') + 'MAVIS' }[screen];
}

function onRoute() {
  const m = location.hash.match(/^#\/doc\/([\w-]+)/);
  S.route = m ? { name: 'doc', id: m[1] } : { name: 'home' };
  S.ui.sidebarOpen = false;
  if (S.route.name === 'doc' && (!S.doc || S.doc.id !== S.route.id)) { S.doc = null; S.ui.selPage = null; }
  if (S.app) { const s = pickScreen(); if (s !== S.screen || s === 'doc') { S.screen = s; mountScreen(s); } renderSidebar(); }
}

/* ------------------------------------------------------------------ sidebar */
function renderSidebar() {
  const sb = $('#sidebar'); if (!S.app) return;
  const sc = S.app.scan; const locked = !S.homeOverride && (sc && ['connecting', 'scanning', 'finalizing'].includes(sc.phase) || S.screen === 'final');
  sb.classList.toggle('open', S.ui.sidebarOpen); $('#scrim').classList.toggle('show', S.ui.sidebarOpen);
  let body;
  if (locked) {
    body = `<div class="side-locked"><b>Scan in progress</b>Your documents are available again once this scan has been processed.</div><div class="history"></div>`;
  } else {
    const items = S.docs.length ? S.docs.map((d) => `
      <button class="hist-item ${S.route.name === 'doc' && S.route.id === d.id ? 'active' : ''}" data-doc="${esc(d.id)}">
        <div class="t">${esc(d.title)}</div>
        <div class="m">${d.page_count} page${d.page_count === 1 ? '' : 's'} &middot; ${esc(fmtDate(d.created_at))}</div>
        ${d.last_question ? `<div class="q">${esc(d.last_question)}</div>` : ''}
      </button>`).join('') : '<div class="hist-empty">Scanned documents and their conversations appear here.</div>';
    body = `<div class="side-actions"><button class="btn primary block" id="newScan">${ico('plus')} New scan</button></div>
      <div class="side-title">History</div><div class="history">${items}</div>`;
  }
  sb.innerHTML = `<a class="brand" id="brandHome" href="#/" title="Go to the home screen">${ico('logo')}<div><b>MAVIS</b><small>AI document assistant</small></div></a>${body}
    <div class="side-foot" id="sysfoot"></div>`;
  renderSidebarStatus();
}
function renderSidebarStatus() {
  const foot = $('#sysfoot'); if (!foot || !S.app) return;
  const { llm, reader } = S.app;
  const o = readerDot(reader);
  const l = llm.status === 'ok' ? ['ok', `${llm.model} ready`] : llm.status === 'checking' ? ['warn pulse', 'Checking AI model'] : ['bad', 'AI model unavailable'];
  foot.innerHTML = `<div class="sys"><span class="dot ${o[0]}"></span><span class="name">${o[1]}</span></div>
    <div class="sys"><span class="dot ${l[0]}"></span><span class="name">${esc(l[1])}</span></div>
    <div class="local-note">${reader.remote ? `${ico('alert')} Page photos go to ${esc(reader.label)}` : `${ico('lock')} Everything runs on this computer`}</div>`;
}
const READER_NAMES = { gemini: 'Gemini', ocrspace: 'OCR.space', local: 'this computer' };
function readerDot(reader) {
  const name = reader.remote ? reader.label : 'Local reader';
  switch (reader.status) {
    case 'ok': return ['ok', `${name} ready`];
    case 'checking': return ['warn pulse', `Checking ${name}`];
    case 'offline': return ['warn', 'Offline: local reader will be used'];
    case 'missing_key': return ['bad', `${reader.label} API key missing`];
    case 'rejected': return ['bad', `${reader.label} key rejected`];
    case 'model_missing': return ['bad', `${reader.label} model not found`];
    default: return ['warn', `${name}: ${reader.message || 'not reachable'}`];
  }
}
function confirmDelete(id, title) {
  const back = document.createElement('div'); back.className = 'modal-back';
  back.innerHTML = `<div class="card modal" role="dialog" aria-modal="true"><h2>Delete this scan?</h2>
    <p><b>${esc(title)}</b> will be removed from this computer together with its page photos and conversation. This can't be undone.</p>
    <div class="actions"><button class="btn" id="keepScan">Keep it</button><button class="btn danger" id="delScan">${ico('trash')} Delete</button></div></div>`;
  document.body.appendChild(back);
  $('#keepScan', back).onclick = () => back.remove(); $('#keepScan', back).focus();
  $('#delScan', back).onclick = async () => {
    $('#delScan', back).disabled = true;
    try { await api('/api/documents/' + id, { method: 'DELETE' }); back.remove(); toast('Scan deleted'); if (S.route.name === 'doc' && S.route.id === id) location.hash = '#/'; await refreshDocs(); if (S.view && S.view.update) S.view.update(); }
    catch (e) { back.remove(); toast(e.message, true); }
  };
}
document.addEventListener('click', (e) => {
  if (e.target.closest('#brandHome')) { e.preventDefault(); goHome(); return; }
  const back = e.target.closest('[data-backtoscan]'); if (back) { backToScan(); return; }
  const del = e.target.closest('[data-del]'); if (del) { confirmDelete(del.dataset.del, del.dataset.title); return; }
  const item = e.target.closest('[data-doc]'); if (item) { location.hash = '#/doc/' + item.dataset.doc; return; }
  if (e.target.closest('#newScan')) { location.hash = '#/'; return; }
  if (e.target.closest('.menu-btn')) { S.ui.sidebarOpen = true; renderSidebar(); return; }
  if (e.target.id === 'scrim') { S.ui.sidebarOpen = false; renderSidebar(); }
  if (!e.target.closest('.menu')) $$('.menu .pop').forEach((p) => { p.hidden = true; });
});

/* ------------------------------------------------------------------ home */
function HomeView(root) {
  const saved = (() => { try { return JSON.parse(localStorage.getItem('mavis.cam') || '{}'); } catch (e) { return {}; } })();
  root.innerHTML = `<div class="home">
    <button class="btn ghost menu-btn" style="margin:-10px 0 10px -10px">${ico('menu')} Menu</button>
    <div id="scanBanner"></div>
    <h1>Scan a document. Then just ask.</h1>
    <p class="lead">Point your phone camera at the pages. MAVIS reads them on this computer, then lets you ask questions and get answers with the page they came from.</p>
    <div class="home-grid">
      <section class="card panel">
        <h2>Connect a camera</h2>
        <p class="sub">Open the DroidCam app on your phone and enter the address it shows.</p>
        <form id="camForm" autocomplete="off">
          <div class="form-row">
            <label class="field">IP address<input class="input" id="ip" inputmode="decimal" placeholder="192.168.1.3" value="${esc(saved.ip || '')}"></label>
            <label class="field">Port<input class="input" id="port" inputmode="numeric" placeholder="4747" value="${esc(saved.port || '4747')}"></label>
          </div>
          <button class="btn primary lg block" id="connect" type="submit">${ico('camera')} Connect and start scanning</button>
          <div class="error-text" id="formError" role="alert"></div>
          <div class="capmode">
            <label class="field">How pages are captured
              <select class="input" id="selector">
                <option value="b0">Smart - when the view changes</option>
                <option value="interval">Timed - one page every few seconds</option>
                <option value="c3">Steady view (experimental)</option>
              </select></label>
            <label class="field" id="intervalWrap" hidden>Every (seconds)
              <input class="input" id="interval" type="number" min="2" max="30" step="1" value="5"></label>
          </div>
          <p class="hint" id="modeHint"></p>
        </form>
        <div class="divider">or</div>
        <button class="btn block" id="demo">${ico('sparkle')} Try it with a simulated camera</button>
      </section>
      <section class="card panel">
        <h2>System check</h2>
        <p class="sub">Everything needed for a scan, at a glance.</p>
        <ul class="checklist" id="checks"></ul>
      </section>
    </div>
    <div class="recent" id="recent"></div>
    <div class="trust"><span id="trustPrivacy">${ico('lock')} Pages never leave this computer</span><span>${ico('quote')} Answers cite the page they came from</span><span>${ico('check')} Quotes are checked against the scanned text</span></div>
  </div>`;
  $('#selector', root).value = saved.selector || 'b0';
  $('#interval', root).value = saved.interval || 5;
  const HINTS = {
    b0: 'Captures a page a few seconds after the view changes and skips repeats of the same view.',
    interval: 'Captures whatever is in view on a fixed timer (the sharpest frame of the half second before each tick). Turn the page right after a capture. Pages held longer than the interval are captured twice.',
    c3: 'Waits until the page is held steady and keeps its sharpest frame. Experimental.',
  };
  const syncMode = () => {
    const m = $('#selector', root).value;
    $('#intervalWrap', root).hidden = m !== 'interval'; $('#modeHint', root).textContent = HINTS[m];
  };
  $('#selector', root).addEventListener('change', syncMode); syncMode();
  const err = (m) => { $('#formError', root).textContent = m || ''; };
  const start = async (body) => {
    err('');
    const btns = $$('#connect, #demo', root); btns.forEach((b) => { b.disabled = true; });
    try { await api('/api/scan/start', { method: 'POST', body }); } catch (e) { err(e.message); btns.forEach((b) => { b.disabled = false; }); }
  };
  $('#camForm', root).addEventListener('submit', (e) => {
    e.preventDefault();
    const ip = $('#ip', root).value.trim(), port = $('#port', root).value.trim(), selector = $('#selector', root).value;
    const interval_s = Number($('#interval', root).value) || 5;
    try { localStorage.setItem('mavis.cam', JSON.stringify({ ip, port, selector, interval: interval_s })); } catch (x) { /* private mode */ }
    start({ kind: 'droidcam', ip, port, selector, interval_s });
  });
  $('#demo', root).addEventListener('click', () => start({ kind: 'demo', selector: $('#selector', root).value, interval_s: Number($('#interval', root).value) || 5 }));
  if (S.notice) { err(S.notice); S.notice = null; }
  return {
    update() {
      const { llm, reader } = S.app;
      const sc = S.app.scan; const busyScan = !!sc && ['connecting', 'scanning', 'finalizing'].includes(sc.phase);
      let banner = '';
      if (sc && busyScan) {
        banner = `<div class="callout info" style="margin:0 0 22px"><span class="spinner" style="margin-top:3px"></span><div style="flex:1"><b>${sc.phase === 'finalizing' ? 'Your scan is being finalized' : 'A scan is in progress'}</b><div>${sc.capture.pages_captured} page${sc.capture.pages_captured === 1 ? '' : 's'} captured. It keeps running while you are here.</div></div><button class="btn primary sm" data-backtoscan>${sc.phase === 'finalizing' ? 'Show progress' : 'Return to scan'}</button></div>`;
      } else if (sc && sc.phase === 'finished' && !sc.empty && !S.handoff[sc.id]) {
        const prep = S.app.prepare[sc.id]; const ready = prep && prep.status === 'ready';
        banner = `<div class="callout ${ready ? 'info' : 'warn'}" style="margin:0 0 22px">${ico(ready ? 'check' : 'alert')}<div style="flex:1"><b>${ready ? 'Your document is ready' : 'Your document is saved'}</b><div>${ready ? 'Open it to ask questions.' : esc((prep && prep.message) || 'The assistant is still getting ready.')}</div></div><button class="btn primary sm" data-backtoscan>${ready ? 'Open document' : 'Show progress'}</button></div>`;
      }
      const sb = $('#scanBanner', root); if (sb.dataset.k !== banner) { sb.innerHTML = banner; sb.dataset.k = banner; }
      $$('#connect, #demo', root).forEach((b) => { b.disabled = busyScan; });
      const gem = !!reader.remote;
      setHTML(root, '#trustPrivacy', gem ? `${ico('alert')} Page photos are sent to ${esc(reader.label)} to be read. Answers are written on this computer.` : `${ico('lock')} Pages never leave this computer`);
      const row = (state, title, why) => `<li><span class="dot ${state}" style="margin-top:7px"></span><span>${title}<span class="why">${why}</span></span></li>`;
      const rd = readerDot(reader);
      const readerWhy = {
        ok: 'Each page photo is sent to Google to be read. If the internet is down, this computer reads the pages instead (slower).',
        checking: 'Checking the connection and the key...',
        offline: 'No internet connection. Pages will be read on this computer (slower) until it is back.',
        missing_key: 'Create a key in Google AI Studio, run  <code>setx GEMINI_API_KEY "..."</code>  in a terminal, then restart MAVIS from a new terminal.',
      }[reader.status] || esc(reader.message || '');
      $('#checks', root).innerHTML =
        row(rd[0], 'Page reader' + (gem ? ': ' + esc(reader.label) + ' (' + esc(reader.model) + ')' : ' (development)'), readerWhy + (['rejected', 'model_missing', 'unreachable', 'offline'].includes(reader.status) ? ' <button class="btn sm" id="recheckReader" type="button" style="margin-top:6px">' + ico('refresh') + ' Check again</button>' : '')) +
        row(llm.status === 'ok' ? 'ok' : llm.status === 'checking' ? 'warn pulse' : 'bad', 'AI assistant (' + esc(llm.model) + ')',
          llm.status === 'ok' ? 'Ready to answer questions about your documents.' : esc(llm.message) + ' <button class="btn sm" id="recheck" type="button" style="margin-top:6px">' + ico('refresh') + ' Check again</button>') +
        row('', 'Camera', 'Not connected yet. Connect one on the left, or use the simulated camera.');
      const r = $('#recheck', root); if (r) r.onclick = async () => { r.disabled = true; try { await api('/api/llm/check', { method: 'POST' }); } catch (e) { toast(e.message, true); } };
      const rr = $('#recheckReader', root); if (rr) rr.onclick = async () => { rr.disabled = true; try { await api('/api/reader/check', { method: 'POST' }); } catch (e) { toast(e.message, true); } };
      const key = JSON.stringify(S.docs.map((d) => [d.id, d.title, d.page_count, d.turn_count]));
      const rec = $('#recent', root);
      if (rec.dataset.k !== key) {
        rec.dataset.k = key;
        rec.innerHTML = S.docs.length ? `<h3>Your scans</h3><div class="recent-grid">${S.docs.map((d) => `
          <div class="recent-card"><button class="rc-open" data-doc="${esc(d.id)}"><b>${esc(d.title)}</b><span>${d.page_count} page${d.page_count === 1 ? '' : 's'} &middot; ${esc(fmtDate(d.created_at))} &middot; ${d.turn_count} question${d.turn_count === 1 ? '' : 's'}</span></button>
          <button class="btn ghost sm rc-del" data-del="${esc(d.id)}" data-title="${esc(d.title)}" aria-label="Delete ${esc(d.title)}" title="Delete this scan">${ico('trash')}</button></div>`).join('')}</div>` : '';
      }
    },
  };
}

/* ------------------------------------------------------------------ scan */
function ScanView(root) {
  root.innerHTML = `<div class="scan">
    <div class="topbar">
      <button class="btn ghost menu-btn">${ico('menu')}</button>
      <h1>Scanning <span class="live-pill" id="livePill"><span class="dot"></span><span id="liveText">CONNECTING</span></span></h1>
      <span class="grow"></span>
      <span class="chip" id="chipSource"></span><span class="chip" id="chipFps" title="Frames per second arriving from the camera / shown in this window"></span><span class="chip" id="chipRes"></span>
      <span class="chip" title="Time since scanning started">${ico('camera')}<span data-since="0" id="elapsed">0.0s</span></span>
      <button class="btn danger" id="stop" disabled>${ico('stop')} Stop scanning</button>
    </div>
    <div id="health"></div>
    <div class="scan-grid">
      <div>
        <div class="stage" id="stage">
          <img class="feed" id="feed" alt="Live camera feed" hidden>
          <div class="ph" id="ph"><div class="spinner" style="width:28px;height:28px"></div><div id="phText">Connecting to the camera&hellip;</div><button class="btn sm" id="cancel" style="margin-top:6px">Cancel</button></div>
          <div class="corners"></div><div class="flash" id="flash"></div>
          <div class="hud"><span class="chip" id="hudPages">Pages: 0</span></div>
          <div class="caption" id="caption" hidden><span class="spinner" id="capSpin"></span><span id="capText"></span></div>
        </div>
        <p class="hint" id="tip" style="margin:8px 4px 0"></p>
        <div class="tiles">
          <div class="card tile"><div class="k">Pages captured</div><div class="v" id="tPages">0</div><div class="s" id="tPagesS">&nbsp;</div></div>
          <div class="card tile"><div class="k">Reading now</div><div class="v" id="tNow">Idle</div><div class="s" id="tNowS">&nbsp;</div></div>
          <div class="card tile"><div class="k">Reading queue</div><div class="v" id="tQueue">0 / 8</div><div class="qbar" id="qbar"></div><div class="s" id="tQueueS">&nbsp;</div></div>
          <div class="card tile"><div class="k">Average read time</div><div class="v" id="tConf">-</div><div class="s" id="tConfS">per page</div></div>
        </div>
      </div>
      <div class="side-col">
        <section class="card feed-card">
          <div class="feed-head"><h2>Pages</h2><span class="chip" id="feedCount">0</span></div>
          <div class="pages" id="pages"></div>
        </section>
        <section class="card"><div class="feed-head"><h2>Activity</h2></div><div class="log" id="log"></div></section>
      </div>
    </div></div>`;
  const stage = $('#stage', root);
  const resize = () => stage.style.setProperty('--h', stage.clientHeight + 'px'); resize(); window.addEventListener('resize', resize);
  $('#cancel', root).onclick = () => api('/api/scan/cancel', { method: 'POST' }).catch((e) => toast(e.message, true));
  $('#stop', root).onclick = () => stopDialog();
  let feedStarted = false; const pageEls = new Map();

  function stopDialog() {
    const sc = S.app.scan; const waiting = sc.ocr.queue_depth + (sc.ocr.busy ? 1 : 0);
    const back = document.createElement('div'); back.className = 'modal-back';
    back.innerHTML = `<div class="card modal" role="dialog" aria-modal="true"><h2>Stop scanning?</h2>
      <p>No new pages will be captured. MAVIS will finish reading the ${sc.capture.pages_captured} page${sc.capture.pages_captured === 1 ? '' : 's'} already captured${waiting ? ` (${waiting} still being read)` : ''}, prepare the document, and then open it for questions. You'll see progress the whole time.</p>
      <div class="actions"><button class="btn" id="keep">Keep scanning</button><button class="btn danger" id="confirm">${ico('stop')} Stop and finish</button></div></div>`;
    document.body.appendChild(back);
    $('#keep', back).onclick = () => back.remove();
    $('#confirm', back).onclick = async () => { back.remove(); try { await api('/api/scan/stop', { method: 'POST' }); } catch (e) { toast(e.message, true); } };
    $('#confirm', back).focus();
  }

  function message(sc) {
    const o = sc.ocr, c = sc.capture;
    if (sc.connection.state === 'lost') return { bad: true, text: 'Connection lost - the camera stopped sending video. You can still stop and process the pages already captured.' };
    if (c.mode === 'interval' && c.next_capture_at) {
      const reading = o.current_page ? `Reading page ${o.current_page}. ` : '';
      return { reading: !!o.current_page, html: `${reading}Next page is captured in <b data-until="${c.next_capture_at}">${c.interval_s}s</b> - hold the page steady` };
    }
    if (o.current_page) return { reading: true, text: `Reading page ${o.current_page}...` };
    if (o.queue_depth) return { reading: true, text: `${o.queue_depth} page${o.queue_depth === 1 ? '' : 's'} waiting to be read` };
    switch (c.status) {
      case 'cooldown': return { text: `Page ${c.pages_captured} captured - move to the next page` };
      case 'duplicate': return { text: 'Same page as before - turn to the next page when you are ready' };
      case 'queue_full': return { warn: true, text: 'Reading queue is full - pause a moment before the next page' };
      case 'watching': return { text: 'Watching for a steady page' };
      default: return { text: c.pages_captured ? 'Waiting for the next page' : 'Point the camera at a page - capture starts automatically' };
    }
  }

  return {
    update() {
      const sc = S.app.scan; if (!sc) return;
      const live = sc.connection.state === 'live';
      const pill = $('#livePill', root); pill.classList.toggle('off', !live);
      setText(root, '#liveText', sc.phase === 'connecting' ? 'CONNECTING' : live ? 'LIVE' : 'OFFLINE');
      setText(root, '#chipSource', (sc.source.kind === 'demo' ? 'Simulated camera' : 'DroidCam ' + sc.source.label));
      setText(root, '#chipFps', live ? `${sc.connection.fps.toFixed(0)} fps in - ${sc.connection.shown_fps.toFixed(0)} shown` : '- fps');
      const slowCam = live && sc.connection.frames > 90 && sc.connection.fps < 12;
      const gemini = !!S.app.reader.remote;   // a cloud API reads the pages
      const slowOcr = !gemini && sc.ocr.avg_page_s && sc.ocr.avg_page_s > 10;
      const slowGem = gemini && sc.ocr.avg_page_s && sc.ocr.avg_page_s > 12;
      const limited = sc.ocr.rate_limited_until && sc.ocr.rate_limited_until > serverNow();
      let health = '';
      if (sc.phase === 'scanning' || sc.phase === 'connecting') {
        if (limited) health = `<div class="callout warn" style="margin:0">${ico('alert')}<div><b>${esc(S.app.reader.label)}'s rate limit was reached.</b><div>Pages are waiting for it to clear (<b data-until="${sc.ocr.rate_limited_until}">...</b>). Nothing is lost; you can keep scanning and every page will be read. MAVIS now sends requests more slowly.</div></div></div>`;
        else if (slowCam || slowOcr) health = `<div class="callout warn" style="margin:0">${ico('alert')}<div><b>This computer is struggling to keep up.</b><div>${slowCam ? `The camera is arriving at only ${sc.connection.fps.toFixed(0)} frames per second, so the live picture will lag. ` : ''}${slowOcr ? `Reading a page is taking ${sc.ocr.avg_page_s}s (about 2 to 4s is normal). ` : ''}Close other heavy programs, keep the laptop plugged in, and make sure nothing else is using the CPU.</div></div></div>`;
        else if (slowGem) health = `<div class="callout info" style="margin:0">${ico('alert')}<div><b>${esc(S.app.reader.label)} is taking ${sc.ocr.avg_page_s}s per page.</b><div>That is the online service, not this computer. The live picture is not affected.</div></div></div>`;
      }
      const hb = $('#health', root); if (hb.dataset.k !== health) { hb.innerHTML = health; hb.dataset.k = health; }
      setText(root, '#tip', sc.connection.size && sc.connection.size[0] < 1600 ? `Camera stream is ${sc.connection.size.join('x')}. For small print, choose 1920x1080 in the DroidCam app, fill the frame with the page and keep the phone parallel to it.` : 'Fill the frame with the page, keep the phone parallel to it and hold it steady.');
      setText(root, '#chipRes', sc.connection.size ? sc.connection.size.join('x') : '-');
      const el = $('#elapsed', root); if (sc.started_at) el.dataset.since = sc.started_at;
      $('#stop', root).disabled = sc.phase !== 'scanning';
      $('#cancel', root).hidden = sc.phase !== 'connecting';
      if (sc.phase === 'scanning' && !feedStarted) { feedStarted = true; const f = $('#feed', root); f.src = '/api/scan/video?t=' + Date.now(); f.hidden = false; $('#ph', root).hidden = true; }
      if (feedStarted && !live) { $('#ph', root).hidden = false; $('#phText', root).textContent = 'The camera stopped sending video.'; $('#ph .spinner', root).hidden = true; }
      const msg = message(sc);
      stage.classList.toggle('reading', !!msg.reading);
      const cap = $('#caption', root); cap.hidden = sc.phase !== 'scanning'; cap.classList.toggle('bad', !!msg.bad);
      $('#capSpin', root).hidden = !msg.reading;
      if (msg.html) { if ($('#capText', root).dataset.h !== msg.html) { $('#capText', root).innerHTML = msg.html; $('#capText', root).dataset.h = msg.html; } }
      else { delete $('#capText', root).dataset.h; setText(root, '#capText', msg.text); }
      if (sc.capture.last_capture_at && sc.capture.last_capture_at !== S.lastCaptureAt) {
        if (S.lastCaptureAt !== null || sc.capture.pages_captured) { const f = $('#flash', root); f.classList.remove('go'); void f.offsetWidth; f.classList.add('go'); }
        S.lastCaptureAt = sc.capture.last_capture_at;
      }
      const pages = sc.pages; const done = pages.filter((p) => p.state === 'done').length; const err = pages.filter((p) => p.state === 'error').length;
      const waiting = pages.length - done - err;
      setText(root, '#hudPages', `Pages: ${pages.length}`);
      setText(root, '#tPages', pages.length); setText(root, '#tPagesS', `${done} read${waiting ? ' - ' + waiting + ' in progress' : ''}${err ? ' - ' + err + ' failed' : ''}`);
      const o = sc.ocr;
      if (o.current_page) {
        $('#tNow', root).textContent = 'Page ' + o.current_page;
        $('#tNowS', root).innerHTML = `<span data-since="${o.current_started_at}">0.0s</span> so far${o.avg_page_s ? ' - avg ' + o.avg_page_s + 's per page' : ''}`;
      } else { setText(root, '#tNow', 'Idle'); setText(root, '#tNowS', o.avg_page_s ? `avg ${o.avg_page_s}s per page` : 'waiting for a page'); }
      setText(root, '#tQueue', `${o.queue_depth} / ${o.queue_max}`);
      setHTML(root, '#qbar', Array.from({ length: o.queue_max }, (_, i) => `<i class="${i < o.queue_depth ? 'on' : ''}"></i>`).join(''));
      setText(root, '#tQueueS', sc.capture.skipped_queue_full ? `${sc.capture.skipped_queue_full} view(s) skipped - queue was full` : 'waiting to be read');
      setText(root, '#tConf', o.avg_page_s ? o.avg_page_s + 's' : '-');
      setText(root, '#feedCount', pages.length);
      reconcilePages(pages);
      setHTML(root, '#log', sc.activity.slice().reverse().map((a) => `<div class="${esc(a.kind)}"><time>${fmtClock(a.t)}</time><span>${esc(a.msg)}</span></div>`).join(''));
    },
  };

  function reconcilePages(pages) {
    const holder = $('#pages', root);
    if (!pages.length) { if (!$('.feed-empty', holder)) holder.innerHTML = '<div class="feed-empty">Pages appear here the moment they are captured, then fill in as they are read.</div>'; return; }
    const empty = $('.feed-empty', holder); if (empty) empty.remove();
    pages.slice().reverse().forEach((p, idx) => {
      let el = pageEls.get(p.n);
      if (!el) { el = document.createElement('div'); el.innerHTML = `<img class="th" alt="" src="/api/scan/pages/${p.n}/image?s=${encodeURIComponent(S.app.scan.id)}"><div class="bd"></div>`; pageEls.set(p.n, el); holder.insertBefore(el, holder.children[idx] || null); }
      el.className = 'page ' + p.state;
      let chip;
      if (p.state === 'queued') chip = '<span class="chip">Waiting to be read</span>';
      else if (p.state === 'reading') chip = `<span class="chip brand"><span class="spinner" style="width:11px;height:11px;border-width:2px"></span>Reading <span data-since="${p.started_at}">0.0s</span></span>`;
      else if (p.state === 'error') chip = '<span class="chip bad">Could not be read</span>';
      else if (p.blank) chip = '<span class="chip warn" title="Nothing readable (for example only the DroidCam watermark). It will not be part of the document.">No text - left out</span>';
      else chip = `<span class="chip ok">${ico('check')} Read</span><span class="chip">${p.chars.toLocaleString()} chars</span>${p.confidence != null ? `<span class="chip" title="Average recognition confidence">${pct(p.confidence)}</span>` : ''}${S.app.reader.remote ? `<span class="chip ${p.reader !== 'local' ? 'info' : 'warn'}" title="${p.reader !== 'local' ? 'Read by ' + esc(READER_NAMES[p.reader] || p.reader) + ' (no confidence score is available)' : 'The online reader was not used for this page; it was read on this computer'}">${p.reader !== 'local' ? esc(READER_NAMES[p.reader] || p.reader) : 'Local (fallback)'}</span>` : ''}${p.low_content ? '<span class="chip warn" title="Almost no text was found - this may be a blurry or empty view">Very little text</span>' : ''}`;
      const html = `<div class="row"><b>Page ${p.n}</b>${chip}</div>${p.state === 'reading' || p.state === 'queued' ? '<div class="progress indet" style="margin-top:7px"><i></i></div>' : `<div class="pv">${esc(p.error || p.preview || '(no text found)')}</div>`}`;
      const bd = $('.bd', el); const key = p.state + '|' + p.chars + '|' + p.confidence + '|' + (p.started_at || '');
      if (el.dataset.key !== key) { bd.innerHTML = html; el.dataset.key = key; }
    });
  }
}

/* ------------------------------------------------------------------ finalizing */
function FinalView(root) {
  root.innerHTML = `<div class="final">
    <button class="btn ghost menu-btn" style="margin:-10px 0 10px -10px">${ico('menu')} Menu</button>
    <h1 id="fTitle">Finalizing your document</h1>
    <p class="lead" id="fLead">Every page you scanned is being read and prepared. Nothing is lost - please keep this window open.</p>
    <section class="card overall"><div class="top"><span class="pct" id="fPct">0%</span><span class="muted" id="fStep">Step 1 of 5</span><span class="grow"></span><span class="chip">${ico('camera')}<span id="fElapsed" data-since="0">0.0s</span></span></div>
      <div class="progress"><i id="fBar" style="width:0%"></i></div>
      <div class="beat"><span class="dot ok pulse" id="fDot"></span><span id="fBeat">Working&hellip;</span></div></section>
    <section class="card steps" id="fSteps"></section>
    <div id="fThumbs" class="thumbs"></div>
    <div id="fCallout"></div></div>`;
  return {
    update() {
      const sc = S.app.scan; if (!sc) return;
      const prep = S.app.prepare[sc.id]; const llm = S.app.llm;
      const pages = sc.pages; const settled = pages.filter((p) => p.state === 'done' || p.state === 'error').length;
      const steps = sc.stages.map((s) => ({ key: s.key, label: s.label, state: s.state, detail: s.detail, started: s.started_at, ended: s.ended_at, frac: s.state === 'done' || s.state === 'warn' ? 1 : s.state === 'active' ? 0.4 : 0, w: 1 }));
      const rp = steps.find((s) => s.key === 'read_pages'); rp.w = 3;
      if (rp.state === 'active') rp.frac = pages.length ? settled / pages.length : 0.5;
      rp.detail = rp.state === 'pending' ? '' : (rp.state === 'active' ? `${settled} of ${pages.length} pages read` : rp.detail);
      const showContext = !sc.empty;
      if (showContext) {
        const ctx = { key: 'context', label: 'Preparing the AI assistant', state: 'pending', detail: 'Starts once the document is saved', frac: 0, w: Math.max(1, Math.min(3, (prep && prep.total) || 1)) };
        if (prep) {
          const map = { queued: 'active', running: 'active', ready: 'done', unavailable: 'warn', failed: 'bad' }; ctx.state = map[prep.status] || 'active';
          ctx.detail = prep.status === 'ready' ? 'Ready for your questions' : prep.message; ctx.started = prep.started_at; ctx.ended = prep.ended_at;
          ctx.frac = prep.status === 'ready' ? 1 : prep.total ? prep.done / prep.total : 0.4; ctx.progress = prep.status === 'running' && prep.total ? [prep.done, prep.total] : null;
          if (prep.status === 'running' && prep.done === 0) ctx.detail = prep.message + ' (the first load can take about a minute)';
        }
        steps.push(ctx);
      }
      const wsum = steps.reduce((a, s) => a + s.w, 0); const frac = steps.reduce((a, s) => a + s.w * Math.min(1, s.frac), 0) / wsum;
      const activeIdx = Math.max(0, steps.findIndex((s) => s.state === 'active'));
      const allDone = steps.every((s) => s.state === 'done' || s.state === 'warn');
      setText(root, '#fPct', Math.round(frac * 100) + '%');
      $('#fBar', root).style.width = Math.round(frac * 100) + '%';
      setText(root, '#fStep', allDone ? 'Complete' : `Step ${activeIdx + 1} of ${steps.length}: ${steps[activeIdx].label}`);
      const fe = $('#fElapsed', root); if (sc.finalize_started_at) fe.dataset.since = sc.finalize_started_at;
      const failed = prep && ['unavailable', 'failed'].includes(prep.status);
      let beat = 'Working...';
      if (sc.phase === 'finalizing' && sc.ocr.rate_limited_until && sc.ocr.rate_limited_until > serverNow()) beat = `Waiting for ${esc(S.app.reader.label)}'s rate limit to clear - <span data-until="${sc.ocr.rate_limited_until}">...</span> (nothing is lost)`;
      else if (sc.phase === 'finalizing' && sc.ocr.current_page) beat = `Reading page ${sc.ocr.current_page} of ${pages.length} - <span data-since="${sc.ocr.current_started_at}">0.0s</span>`;
      else if (sc.phase === 'finalizing') beat = steps[activeIdx].label + '...';
      else if (prep && prep.status === 'running') beat = `${esc(prep.message)} - <span data-since="${prep.started_at}">0.0s</span>`;
      if (sc.empty) beat = 'Finished';
      if (allDone || failed) beat = 'Finished';
      $('#fBeat', root).innerHTML = beat; $('#fDot', root).className = 'dot ' + (allDone || failed ? (failed ? 'warn' : 'ok') : 'ok pulse');
      setHTML(root, '#fSteps', steps.map((s, i) => {
        const icon = s.state === 'done' ? ico('check') : s.state === 'active' ? '<span class="spinner" style="width:14px;height:14px"></span>' : s.state === 'warn' ? ico('alert') : s.state === 'bad' ? ico('x') : i + 1;
        const bar = s.state === 'active' ? (s.key === 'read_pages' && pages.length ? `<div class="progress"><i style="width:${Math.round(s.frac * 100)}%"></i></div>` : s.progress ? `<div class="progress"><i style="width:${Math.round(s.progress[0] / s.progress[1] * 100)}%"></i></div>` : '<div class="progress indet"><i></i></div>') : '';
        const dur = s.started ? ((s.ended || serverNow()) - s.started) : null;
        return `<div class="step ${s.state}"><div class="ic">${icon}</div><div><div class="nm">${esc(s.label)}</div><div class="dt">${esc(s.detail || '')}</div>${bar}</div><div class="el">${dur != null ? (s.state === 'active' ? `<span data-since="${s.started}">0.0s</span>` : fmtDur(dur)) : ''}</div></div>`;
      }).join(''));
      setHTML(root, '#fThumbs', pages.map((p) => `<div class="tb ${p.state}"><img alt="" src="/api/scan/pages/${p.n}/image?s=${encodeURIComponent(S.app.scan.id)}"><span>${p.n}</span></div>`).join(''));
      let callout = '';
      if (sc.empty) callout = `<div class="callout warn">${ico('alert')}<div><b>No readable pages were captured.</b><div>Nothing was saved. Make sure the page fills the camera view and is well lit, then try again.<div class="actions"><button class="btn primary" id="again">Start a new scan</button></div></div></div></div>`;
      else if (failed) callout = `<div class="callout warn">${ico('alert')}<div><b>Your document is saved, but the AI assistant isn't available.</b><div>${esc(prep.message)}</div><div class="actions"><button class="btn primary" id="openDoc">Open the document</button><button class="btn" id="retry">${ico('refresh')} Retry</button></div></div></div>`;
      const co = $('#fCallout', root); if (co.dataset.k !== callout) { co.innerHTML = callout; co.dataset.k = callout; }
      const bind = (id, fn) => { const b = $(id, root); if (b) b.onclick = fn; };
      bind('#again', async () => { await api('/api/scan/dismiss', { method: 'POST' }).catch(() => {}); location.hash = '#/'; S.screen = null; onState(); });
      bind('#openDoc', async () => { S.handoff[sc.id] = true; await api('/api/scan/dismiss', { method: 'POST' }).catch(() => {}); location.hash = '#/doc/' + sc.id; });
      bind('#retry', async () => { try { await api('/api/llm/check', { method: 'POST' }); await api(`/api/documents/${sc.id}/prepare`, { method: 'POST' }); } catch (e) { toast(e.message, true); } });
      setText(root, '#fTitle', allDone ? 'Your document is ready' : sc.empty ? 'Nothing to finalize' : 'Finalizing your document');
    },
  };
}

/* ------------------------------------------------------------------ workspace */
function renderMd(text) {
  const lines = esc(text || '').split(/\r?\n/); let html = '', list = false;
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  for (const raw of lines) {
    const m = raw.match(/^\s*(?:[-*•])\s+(.*)$/);
    if (m) { if (!list) { html += '<ul>'; list = true; } html += `<li>${inline(m[1])}</li>`; continue; }
    if (list) { html += '</ul>'; list = false; }
    if (raw.trim()) html += `<p>${inline(raw)}</p>`;
  }
  return html + (list ? '</ul>' : '');
}
const SOURCE = {
  document: ['ok', 'check', 'From your document - verified'],
  document_unverified: ['warn', 'alert', 'From your document - please check the sources'],
  not_in_document: ['', 'doc', 'Not found in the document'],
  general_knowledge: ['info', 'sparkle', 'General knowledge - not from your document'],
};
function turnHTML(t) {
  const r = t.result || {}; let ai;
  if (r.error) {
    ai = `<div class="callout bad" style="margin:0">${ico('alert')}<div><b>That didn't work.</b><div>${esc(r.error.message)}</div></div></div>`;
  } else {
    const src = SOURCE[r.source] || SOURCE.not_in_document;
    const cites = (r.evidence || []).map((e) => {
      const st = e.match === 'exact' ? ['ok', 'matches the scanned text'] : e.match === 'approx' ? ['warn', 'approximate match'] : ['bad', 'not found in the scanned text'];
      return `<button class="cite" data-page="${e.page}" data-quote="${esc(e.quote)}"><span class="pg">Page ${esc(e.page)}</span><span><q>${esc(e.quote)}</q><span class="st ${st[0]}">${st[1]}</span></span></button>`;
    }).join('');
    const nWarn = (r.warnings || []).length;
    const notes = nWarn ? `<details class="notes" ${r.source === 'document' ? '' : 'open'}><summary><b>${r.source === 'document' ? `${nWarn} note${nWarn === 1 ? '' : 's'} about reading quality` : 'Worth double-checking'}</b></summary><ul>${r.warnings.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></details>` : '';
    const gk = r.general_knowledge ? `<div class="gk"><b>Not from your document</b>${esc(r.general_knowledge)}</div>` : '';
    ai = `<div class="src"><span class="chip ${src[0]}">${ico(src[1])} ${src[2]}</span></div>
      <div class="answer">${renderMd(r.answer || (r.source === 'not_in_document' ? "The scanned pages don't contain this." : ''))}</div>
      ${cites ? `<div class="cites">${cites}</div>` : ''}${gk}${notes}
      <div class="ai-meta">${esc(r.model || '')}${r.timing && r.timing.total_s ? ' &middot; ' + r.timing.total_s + 's' : ''} &middot; answered on this computer</div>`;
  }
  return `<div class="msg user"><div class="bubble">${esc(t.question)}</div></div><div class="msg ai"><div class="avatar">${ico('sparkle')}</div><div class="ai-body">${ai}</div></div>`;
}

function DocView(root) {
  const id = S.route.id;
  root.innerHTML = `<div class="ws" id="ws"><section class="chat">
      <div class="ws-head"><button class="btn ghost menu-btn">${ico('menu')}</button><div class="ttl"><h1><input id="title" aria-label="Document title" value=""></h1><div class="meta" id="meta"></div></div>
        <div class="menu"><button class="btn sm" id="exportBtn">${ico('download')} Export</button><div class="pop" id="exportPop" hidden></div></div>
        <button class="btn sm docpane-toggle" id="paneBtn">${ico('panel')} Pages</button></div>
      <div id="banner"></div>
      <div class="thread" id="thread"><div class="thread-inner" id="threadInner"></div></div>
      <div class="composer-wrap"><div class="composer"><textarea id="q" rows="1" placeholder="Ask about this document&hellip;" aria-label="Ask a question"></textarea>
        <button class="btn sm" id="sum" title="Summarize the whole document">${ico('sparkle')} Summarize</button><button class="btn primary send" id="send" aria-label="Send">${ico('send')}</button></div>
        <div class="composer-note" id="note"></div><div class="readby" id="readby"></div></div></section>
    <aside class="docpane" id="docpane"><header><h2>Scanned pages</h2><button class="btn ghost sm" id="paneClose" aria-label="Close">${ico('x')}</button></header><div class="pagelist" id="pagelist"></div></aside></div>`;
  let lastTurns = -1, lastBusy = null;
  const q = $('#q', root);
  q.addEventListener('input', () => { q.style.height = 'auto'; q.style.height = Math.min(q.scrollHeight, 160) + 'px'; });
  q.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send('answer'); } });
  $('#send', root).onclick = () => send('answer'); $('#sum', root).onclick = () => send('summarize');
  $('#paneBtn', root).onclick = () => { $('#docpane', root).classList.add('open'); }; $('#paneClose', root).onclick = () => { $('#docpane', root).classList.remove('open'); };
  $('#exportBtn', root).onclick = (e) => { e.stopPropagation(); const p = $('#exportPop', root); p.hidden = !p.hidden; };
  $('#title', root).addEventListener('change', async (e) => { try { await api('/api/documents/' + id, { method: 'PATCH', body: { title: e.target.value } }); } catch (x) { toast(x.message, true); } });
  root.addEventListener('click', (e) => { const c = e.target.closest('.cite'); if (c) openCitation(Number(c.dataset.page), c.dataset.quote); const s = e.target.closest('[data-suggest]'); if (s) send(s.dataset.mode, s.dataset.suggest); });

  async function load() {
    let fetched;
    try { fetched = await api('/api/documents/' + id); } catch (e) { toast(e.message, true); if (S.route.id === id) location.hash = '#/'; return; }
    if (S.screen !== 'doc' || S.route.id !== id) return;        // the user has gone elsewhere while this loaded
    S.doc = fetched;
    document.title = S.doc.title + ' - MAVIS';
    $('#title', root).value = S.doc.title;
    const d = S.doc;
    $('#meta', root).innerHTML = `<span class="chip">${d.page_count} page${d.page_count === 1 ? '' : 's'}</span>${d.mean_confidence != null ? `<span class="chip" title="Average text-recognition confidence across pages">Reading confidence ${pct(d.mean_confidence)}</span>` : ''}${d.pages_failed.length ? `<span class="chip bad" title="Open the pages panel and use Read this page again">${d.pages_failed.length} page(s) could not be read</span>` : ''}${(d.blank_pages || []).length ? `<span class="chip" title="Captured views with no readable text (for example only the DroidCam watermark) were left out">${d.blank_pages.length} blank view(s) left out</span>` : ''}`;
    const remote = (d.readers || []).filter((r) => r !== 'local').map((r) => READER_NAMES[r] || r);
    $('#readby', root).textContent = remote.length ? `Pages read by ${remote.join(' + ')}${d.readers.includes('local') ? ' and on this computer' : ''}; their photos were sent there.` : 'Pages read on this computer; nothing left it.';
    const pop = $('#exportPop', root); const base = '/api/documents/' + id + '/export/';
    pop.innerHTML = `<a href="${base}text" download>${ico('doc')} Text (.txt)</a><a href="${base}corrected" download>${ico('doc')} Cleaned text (JSON)</a><a href="${base}raw" download>${ico('doc')} Original OCR text (JSON)</a><a href="${base}metadata" download>${ico('doc')} Details (JSON)</a>`;
    if (S.ui.selPage == null && d.pages.length) { const first = d.pages.find((p) => p.status === 'failed') || d.pages[0]; S.ui.selPage = first.n; }
    renderPages(); renderThread(); update();
    const pr = S.app.prepare[id];
    if ((!pr || pr.status === 'cancelled') && !S.app.scan) api(`/api/documents/${id}/prepare`, { method: 'POST' }).catch(() => {});
  }

  function renderThread() {
    const d = S.doc; if (!d) return; const inner = $('#threadInner', root);
    const blocked = composerState().blocked;
    const inFlight = (S.ui.pending && S.ui.pending.docId === id) || (S.app.busy && S.app.busy.doc_id === id);
    let html = d.turns.length || inFlight ? d.turns.map(turnHTML).join('') : `<div class="empty-chat"><div class="orb">${ico('sparkle')}</div><h2>Ask anything about this document</h2>
      <p>Answers come only from the ${d.page_count} page${d.page_count === 1 ? '' : 's'} you scanned, and every quote is checked against the scanned text.</p>
      <div class="suggest"><button data-mode="summarize" data-suggest="" ${blocked ? 'disabled' : ''}>Summarize this document</button><button data-mode="answer" data-suggest="What are the key points?" ${blocked ? 'disabled' : ''}>What are the key points?</button><button data-mode="answer" data-suggest="List any dates, names and amounts mentioned." ${blocked ? 'disabled' : ''}>Dates, names and amounts</button></div></div>`;
    if (S.ui.pending && S.ui.pending.docId === id) html += pendingHTML(S.ui.pending.label, S.ui.pending.started);
    else if (S.app.busy && S.app.busy.doc_id === id) html += pendingHTML(S.app.busy.mode === 'summarize' ? 'Summarize this document' : S.app.busy.question, S.app.busy.started_at);
    inner.innerHTML = html; lastTurns = d.turns.length; lastBusy = html.includes('thinking');
    const th = $('#thread', root); th.scrollTop = th.scrollHeight;
  }
  const pendingHTML = (label, started) => `<div class="msg user"><div class="bubble">${esc(label)}</div></div><div class="msg ai"><div class="avatar">${ico('sparkle')}</div><div class="ai-body"><div class="thinking"><span class="dots"><i></i><i></i><i></i></span><span>Reading your document and writing an answer - <span data-since="${started}">0.0s</span></span></div><div class="ai-meta">This runs on your computer, so it usually takes 10 to 30 seconds.</div></div></div>`;

  function composerState() {
    const app = S.app, llm = app.llm, prep = app.prepare[id]; const sc = app.scan;
    if (sc && ['connecting', 'scanning', 'finalizing'].includes(sc.phase)) return { blocked: true, why: 'A scan is in progress. Questions open when it has finished.' };
    if (llm.status !== 'ok' && llm.status !== 'checking') return { blocked: true, why: 'The AI assistant is not available - see the notice above.', banner: 'llm' };
    if (prep && ['queued', 'running'].includes(prep.status)) return { blocked: true, why: `Preparing the document${prep.total ? ` - page ${Math.min(prep.done + 1, prep.total)} of ${prep.total}` : ''}...`, banner: 'prep', prep };
    if (app.busy && app.busy.doc_id !== id) return { blocked: true, why: 'The assistant is still answering a question about another document. Wait for it, or start a new scan (that stops it).' };
    if ((S.ui.pending && S.ui.pending.docId === id) || app.busy) return { blocked: true, why: 'Thinking...' };
    if (prep && ['unavailable', 'failed'].includes(prep.status)) return { blocked: true, why: prep.message, banner: 'llm' };
    return { blocked: false, why: 'Answers are generated locally and checked against the scanned text. Press Enter to send.' };
  }

  function update() {
    if (!S.doc) { if (!S.docLoading) { S.docLoading = load().finally(() => { S.docLoading = null; }); } return; }
    const cs = composerState(); const app = S.app;
    $('#send', root).disabled = cs.blocked; $('#sum', root).disabled = cs.blocked;
    setText(root, '#note', cs.why);
    let banner = '';
    if (cs.banner === 'llm') banner = `<div class="callout warn banner">${ico('alert')}<div><b>The AI assistant isn't available right now.</b><div>${esc(app.llm.status !== 'ok' ? app.llm.message : (app.prepare[id] || {}).message || '')} You can still read the scanned pages and export the text.</div><div class="actions"><button class="btn sm primary" id="recheck">${ico('refresh')} Check again</button></div></div></div>`;
    else if (cs.banner === 'prep') banner = `<div class="callout info banner"><span class="spinner" style="margin-top:3px"></span><div><b>Getting the assistant ready</b><div>${esc(cs.prep.message)}${cs.prep.done === 0 ? ' - the first start of the model can take about a minute.' : ''}</div></div></div>`;
    const b = $('#banner', root); if (b.dataset.k !== banner) { b.innerHTML = banner; b.dataset.k = banner; const r = $('#recheck', b); if (r) r.onclick = async () => { r.disabled = true; try { await api('/api/llm/check', { method: 'POST' }); await api(`/api/documents/${id}/prepare`, { method: 'POST' }); } catch (e) { toast(e.message, true); r.disabled = false; } }; }
    const busyNow = (S.ui.pending && S.ui.pending.docId === id) || (app.busy && app.busy.doc_id === id);
    if ((S.doc.turns.length !== lastTurns) || (!!busyNow !== lastBusy)) renderThread();
    $$('.suggest button', root).forEach((s) => { s.disabled = cs.blocked; });
  }

  async function send(mode, preset) {
    const text = mode === 'summarize' ? '' : (preset || q.value).trim();
    if (mode === 'answer' && !text) return;
    if (composerState().blocked) return;
    const doc = S.doc;                       // the document this question belongs to, whatever the user opens meanwhile
    S.ui.pending = { docId: id, label: mode === 'summarize' ? 'Summarize this document' : text, started: serverNow() };
    q.value = ''; q.style.height = 'auto'; renderThread(); update();
    const here = () => S.screen === 'doc' && S.route.name === 'doc' && S.route.id === id && S.doc === doc;
    try {
      const { turn } = await api(`/api/documents/${id}/ask`, { method: 'POST', body: { mode, question: text } });
      doc.turns.push(turn); refreshDocs();   // never S.doc: that may be a different document by now
    } catch (e) { toast(e.message, true); if (mode === 'answer' && here()) q.value = text; }
    if (S.ui.pending && S.ui.pending.docId === id) S.ui.pending = null;
    if (here()) { renderThread(); update(); }
  }

  function renderPages() {
    const d = S.doc; if (!d) return; const list = $('#pagelist', root);
    // Every captured page is listed. A page that failed or came out blank still shows its photo and why, with a way to read it again.
    const img = (p) => `/api/documents/${esc(d.id)}/pages/${p.n}/image`;
    list.innerHTML = d.pages.map((p) => {
      const sel = S.ui.selPage === p.n;
      const label = p.label ? `<span class="chip ${p.label === 'HIGH' ? 'ok' : p.label === 'LOW' ? 'bad' : 'warn'}" title="Heuristic text-quality estimate">${p.label.toLowerCase()} quality</span>` : '';
      const status = p.status === 'failed' ? '<span class="chip bad">Could not be read</span>' : p.status === 'blank' ? '<span class="chip warn">No text found</span>' : '';
      const tabs = (names) => `<div class="seg" role="tablist">${names.map(([k, t]) => `<button data-tab="${k}" class="${S.ui.tab === k || (!names.some(([x]) => x === S.ui.tab) && k === names[0][0]) ? 'on' : ''}">${t}</button>`).join('')}</div>`;
      let body = '';
      if (sel && p.status === 'ok') {
        const names = [['ocr', 'Scanned text'], ['clean', 'Cleaned']].concat(p.has_image ? [['image', 'Photo']] : []);
        const tab = names.some(([k]) => k === S.ui.tab) ? S.ui.tab : 'ocr';
        body = `<div class="pbody">${tabs(names)}${tab === 'image' ? `<img alt="Captured page ${p.n}" src="${img(p)}">` : `<div class="ptext" id="ptext">${esc(tab === 'clean' ? p.corrected_text : p.raw_text) || '(no text found)'}</div>`}</div>`;
      } else if (sel) {
        const why = p.status === 'failed' ? `<div class="callout bad" style="margin:0"><div><b>This page could not be read.</b><div>${esc(p.error || 'Unknown reason.')}</div></div></div>` : '<div class="callout warn" style="margin:0"><div><b>No text was found on this view.</b><div>It may be blurry, empty or only the camera app\'s overlay. It is not part of the document.</div></div></div>';
        body = `<div class="pbody">${why}<button class="btn sm primary" data-reread="${p.n}">${ico('refresh')} Read this page again</button>${p.has_image ? `<img alt="Captured page ${p.n}" src="${img(p)}">` : ''}</div>`;
      }
      return `<div class="pcard ${sel ? 'sel' : ''} ${p.status}" data-n="${p.n}"><button class="ph" data-sel="${p.n}">${p.has_image ? `<img class="th" alt="" loading="lazy" src="${img(p)}">` : ''}<span class="pm"><b>Page ${p.n}</b><div>${status}${label}${p.ocr_confidence != null ? `<span class="chip" title="Average recognition confidence">${pct(p.ocr_confidence)} read</span>` : ''}${p.low_content ? '<span class="chip warn">very little text</span>' : ''}</div></span></button>${body}</div>`;
    }).join('');
  }
  async function rereadPage(n, btn) {
    btn.disabled = true; btn.innerHTML = '<span class="spinner" style="width:13px;height:13px"></span> Reading...';
    try {
      const r = await api(`/api/documents/${id}/pages/${n}/reread`, { method: 'POST' });
      toast(r.status === 'ok' ? `Page ${n} was read and added to the document` : `Page ${n} still has no readable text`, r.status !== 'ok');
      S.ui.selPage = n; await load(); S.ui.selPage = n; renderPages(); refreshDocs();     // load() also refreshes the header chips
    } catch (e) { toast(e.message, true); btn.disabled = false; btn.innerHTML = `${ico('refresh')} Read this page again`; }
  }
  root.addEventListener('click', (e) => {
    const rr = e.target.closest('[data-reread]'); if (rr) { rereadPage(Number(rr.dataset.reread), rr); return; }
    const sel = e.target.closest('[data-sel]'); if (sel) { const n = Number(sel.dataset.sel); S.ui.selPage = S.ui.selPage === n ? null : n; renderPages(); return; }
    const tab = e.target.closest('[data-tab]'); if (tab) { S.ui.tab = tab.dataset.tab; renderPages(); }
  });
  function openCitation(page, quote) {
    if (!S.doc) return; S.ui.selPage = page; S.ui.tab = 'ocr'; $('#docpane', root).classList.add('open'); renderPages();
    const p = S.doc.pages.find((x) => x.n === page); const box = $('#ptext', root); if (!p || !box) return;
    const words = String(quote).trim().split(/\s+/).filter(Boolean); const re = (n) => new RegExp(words.slice(0, n).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+'), 'i');
    let m = null; for (const n of [words.length, Math.min(words.length, 8), Math.min(words.length, 4)]) { if (n > 0) { m = re(n).exec(p.raw_text); if (m) break; } }
    if (m) {
      const t = p.raw_text; box.innerHTML = esc(t.slice(0, m.index)) + '<mark class="flashmark">' + esc(m[0]) + '</mark>' + esc(t.slice(m.index + m[0].length));
      const mk = $('mark', box); if (mk) box.scrollTop = Math.max(0, mk.offsetTop - 60);
    } else toast('That quote differs slightly from the scanned text, so it could not be highlighted.');
    const card = $(`.pcard[data-n="${page}"]`, root); if (card) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  return { update };
}

/* ------------------------------------------------------------------ clock for elapsed timers */
setInterval(() => {
  const now = serverNow();
  $$('[data-since]').forEach((el) => { const s = parseFloat(el.dataset.since); if (s > 0) { const t = fmtDur(now - s); if (el.textContent !== t) el.textContent = t; } });
  $$('[data-until]').forEach((el) => { const u = parseFloat(el.dataset.until); if (u > 0) { const t = Math.max(0, u - now).toFixed(1) + 's'; if (el.textContent !== t) el.textContent = t; } });
}, 200);

window.addEventListener('hashchange', onRoute);
connectEvents();
onRoute();
