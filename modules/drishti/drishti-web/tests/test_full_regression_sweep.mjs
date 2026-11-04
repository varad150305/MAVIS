const CDP = 'http://localhost:9333';

async function newTab(url) {
  const res = await fetch(`${CDP}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
  return res.json();
}
function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.addEventListener('open', () => resolve(ws));
    ws.addEventListener('error', reject);
  });
}
let msgId = 1;
function send(ws, method, params = {}) {
  return new Promise((resolve) => {
    const id = msgId++;
    const handler = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.id === id) { ws.removeEventListener('message', handler); resolve(data.result); }
    };
    ws.addEventListener('message', handler);
    ws.send(JSON.stringify({ id, method, params }));
  });
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

const issues = [];
const evalLog = [];

function withTimeout(promise, ms, label){
  return Promise.race([
    promise,
    new Promise((_,reject)=>setTimeout(()=>reject(new Error(`TIMEOUT after ${ms}ms: ${label}`)), ms))
  ]);
}
async function ev(ws, label, expression) {
  console.log(`[running] ${label}...`);
  const r = await withTimeout(
    send(ws, 'Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }),
    15000, label
  );
  const val = r.result?.value;
  const exc = r.exceptionDetails;
  if (exc) {
    issues.push(`[${label}] THREW: ${exc.exception?.description || exc.text}`);
  }
  evalLog.push(`[${label}] -> ${JSON.stringify(val)}`);
  console.log(`[done] ${label} -> ${JSON.stringify(val)}`);
  return val;
}

async function main() {
  const tab = await newTab('about:blank');
  const ws = await connect(tab.webSocketDebuggerUrl);

  ws.addEventListener('message', (msg) => {
    const data = JSON.parse(msg.data);
    if (data.method === 'Runtime.exceptionThrown') {
      const ex = data.params.exceptionDetails;
      issues.push(`[uncaught] ${ex.text} :: ${ex.exception?.description || ''}`);
    }
    if (data.method === 'Runtime.consoleAPICalled' && data.params.type === 'error') {
      const args = data.params.args.map(a => a.value ?? a.description ?? '').join(' ');
      issues.push(`[console.error] ${args}`);
    }
    // Auto-dismiss any native dialogs (alert/confirm/prompt) so nothing hangs
    if (data.method === 'Page.javascriptDialogOpening') {
      send(ws, 'Page.handleJavaScriptDialog', { accept: true, promptText: 'test' });
    }
    // Serve canned Groq responses so mode-specific success-path code (showSafetyBadge,
    // showCrossingBadge, parsePrices, etc.) actually runs instead of dying at 401.
    //
    // IMPORTANT: a cross-origin POST with a JSON Content-Type triggers a CORS
    // preflight OPTIONS request FIRST — Fetch.requestPaused fires for that one
    // before it ever fires for the real POST. Fulfilling the OPTIONS without
    // proper Access-Control-Allow-* headers makes the browser silently block the
    // real POST ("Failed to fetch"), so every mocked test looks like it "passed"
    // while actually only exercising the network-error catch path the whole time
    // — this bit us for real during this session, hence the comment.
    if (data.method === 'Fetch.requestPaused') {
      const { requestId, request } = data.params;
      if (request.url.includes('api.groq.com') && request.method === 'OPTIONS') {
        send(ws, 'Fetch.fulfillRequest', {
          requestId, responseCode: 204,
          responseHeaders: [
            { name: 'Access-Control-Allow-Origin', value: '*' },
            { name: 'Access-Control-Allow-Methods', value: 'POST, OPTIONS' },
            { name: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' }
          ]
        });
      } else if (request.url.includes('api.groq.com')) {
        const body = JSON.stringify({
          choices: [{ message: { content: 'SAFE\nGO\nA person is standing 2 meters ahead near a table. Amazon ₹499. Flipkart ₹520.' } }]
        });
        send(ws, 'Fetch.fulfillRequest', {
          requestId,
          responseCode: 200,
          responseHeaders: [
            { name: 'Content-Type', value: 'application/json' },
            { name: 'Access-Control-Allow-Origin', value: '*' }
          ],
          body: Buffer.from(body).toString('base64')
        });
      } else {
        send(ws, 'Fetch.continueRequest', { requestId });
      }
    }
  });

  await send(ws, 'Runtime.enable');
  await send(ws, 'Network.enable');
  await send(ws, 'Page.enable');
  await send(ws, 'Fetch.enable', { patterns: [{ urlPattern: 'https://api.groq.com/*' }] });
  await send(ws, 'Browser.grantPermissions', {
    permissions: ['geolocation', 'clipboardReadWrite', 'clipboardSanitizedWrite'],
    origin: 'http://localhost:5000'
  });
  await send(ws, 'Emulation.setGeolocationOverride', { latitude: 28.6139, longitude: 77.2090, accuracy: 10 });

  await send(ws, 'Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);

  await ev(ws, 'setup', `
    localStorage.setItem('dk_groq','gsk_fake_test_key');
    localStorage.setItem('dk_sos','+919999999999');
    localStorage.setItem('dk_ai_mode','HYBRID');
    location.reload();
  `);
  await sleep(2000);

  await ev(ws, 'startCam', `startCam()`);
  await sleep(1500);

  // Exercise every mode's setMode + snap path (Groq call is intercepted, see above)
  const modes = ['describe','text','currency','color','people','navigate','objects','prices','food','safety','crossing','translate','gesture'];
  for (const m of modes) {
    await ev(ws, `setMode(${m})`, `setMode('${m}')`);
    await ev(ws, `snap() in ${m}`, `snap()`);
  }

  // Smart Live
  await ev(ws, 'toggleAuto on', `toggleAuto()`);
  await sleep(500);
  await ev(ws, 'toggleAuto off', `toggleAuto()`);

  // Guardian
  await ev(ws, 'toggleGuardian on', `toggleGuardian()`);
  await sleep(500);
  await ev(ws, 'toggleGuardian off', `toggleGuardian()`);

  // Accessibility toggles
  await ev(ws, 'toggleWakeWord on', `toggleWakeWord()`);
  await ev(ws, 'toggleLargeMode on', `toggleLargeMode()`);
  await ev(ws, 'toggleLargeMode off', `toggleLargeMode()`);
  await ev(ws, 'toggleWakeWord off', `toggleWakeWord()`);

  // Voice command engine (bypass real STT — call the parser directly)
  await ev(ws, 'handleCmd describe', `handleCmd('describe')`);
  await sleep(200);
  await ev(ws, 'handleCmd read this', `(function(){ lastCmdTime=0; return handleCmd('read this'); })()`);
  await sleep(200);
  await ev(ws, 'handleCmd navigate to place', `(function(){ lastCmdTime=0; return handleCmd('navigate to connaught place'); })()`);
  await sleep(1500);
  await ev(ws, 'handleCmd stop navigation', `(function(){ lastCmdTime=0; return handleCmd('stop navigation'); })()`);

  // History / chat / sharing
  await ev(ws, 'showHistory', `showHistory()`);
  await ev(ws, 'closeHistory', `closeHistory()`);
  await ev(ws, 'copyDesc', `copyDesc()`);
  await ev(ws, 'copyText', `copyText()`);
  await ev(ws, 'shareDescWA', `shareDescWA()`);
  await ev(ws, 'send (chat ask)', `(function(){ document.getElementById('chatInp').value='is it safe here?'; return send(); })()`);
  await sleep(500);

  // SOS (dialog auto-accepted if it appears; sosNumber pre-set to skip prompt anyway)
  await ev(ws, 'triggerSOS', `triggerSOS()`);
  await sleep(500);

  // Phone modal
  await ev(ws, 'showPhoneModal', `showPhoneModal()`);
  await ev(ws, 'closePhoneModal', `closePhoneModal()`);

  // Mic toggle (just checking it doesn't throw synchronously)
  await ev(ws, 'toggleMic', `toggleMic()`);
  await sleep(300);
  await ev(ws, 'toggleMic off', `toggleMic()`);

  await ev(ws, 'stopCam', `stopCam()`);

  console.log('=== ISSUES FOUND ===');
  if (issues.length === 0) console.log('(none)');
  issues.forEach(i => console.log(i));
  console.log('=== FULL EVAL LOG ===');
  evalLog.forEach(l => console.log(l));

  process.exit(0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
