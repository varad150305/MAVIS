// Verifies the LOCAL_ONLY / HYBRID / CLOUD_ENHANCED switch (ARCHITECTURE_AUDIT.md
// / migration Phase 11) actually gates behavior, not just that the UI exists.
const CDP = 'http://localhost:9333';
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

async function main(){
  const tab = await (await fetch(CDP+'/json/new?about:blank',{method:'PUT'})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r=>ws.addEventListener('open',r));
  let id=1;
  function send(method,params={}){
    return new Promise(resolve=>{
      const myId=id++;
      const h=(ev)=>{const d=JSON.parse(ev.data); if(d.id===myId){ws.removeEventListener('message',h); resolve(d.result);}};
      ws.addEventListener('message',h);
      ws.send(JSON.stringify({id:myId,method,params}));
    });
  }
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Network.enable');
  await send('Fetch.enable', { patterns: [{ urlPattern: 'https://api.groq.com/*' }] });
  let groqWasCalled = false;
  ws.addEventListener('message', (msg) => {
    const data = JSON.parse(msg.data);
    if (data.method === 'Fetch.requestPaused') {
      const { requestId, request } = data.params;
      if (request.url.includes('api.groq.com') && request.method === 'OPTIONS') {
        send('Fetch.fulfillRequest', { requestId, responseCode: 204, responseHeaders: [
          { name: 'Access-Control-Allow-Origin', value: '*' },
          { name: 'Access-Control-Allow-Methods', value: 'POST, OPTIONS' },
          { name: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' }
        ]});
      } else if (request.url.includes('api.groq.com')) {
        groqWasCalled = true;
        const body = JSON.stringify({ choices: [{ message: { content: 'A blue chair.' } }] });
        send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [
          { name: 'Content-Type', value: 'application/json' },
          { name: 'Access-Control-Allow-Origin', value: '*' }
        ], body: Buffer.from(body).toString('base64') });
      } else { send('Fetch.continueRequest', { requestId }); }
    }
  });

  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);
  await send('Runtime.evaluate', { expression: `localStorage.setItem('dk_groq','gsk_fake_test_key'); localStorage.setItem('dk_ai_mode','HYBRID'); location.reload();` });
  await sleep(2000);
  await send('Runtime.evaluate', { expression: `startCam()`, awaitPromise: true });
  await sleep(1000);

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  // LOCAL_ONLY: describe mode used to have no local implementation at all and
  // always failed with a clear "needs cloud" message (see IMPLEMENTATION_PROGRESS.md
  // Entry 30) — as of Entry 30, LOCAL_ONLY routes describe mode through a real
  // on-device vision-language model (callLocalVLM()) instead of refusing outright.
  // Either it genuinely succeeds with a real local description, or — realistically,
  // in this fast/offline test environment which has no time to download the
  // ~1-2GB model — it fails with an honest, specific "local vision model" message.
  // Either way it must still NEVER call the cloud and NEVER fail silently.
  groqWasCalled = false;
  const localOnlyDescribe = await send('Runtime.evaluate', {
    expression: `(async()=>{ document.getElementById('aiModeSel').value='LOCAL_ONLY'; saveAiMode(); setMode('describe'); await snap(); return document.getElementById('stxt').textContent; })()`,
    awaitPromise: true, returnByValue: true
  });
  assert('LOCAL_ONLY + describe mode: does not call the cloud at all', !groqWasCalled, 'groqWasCalled='+groqWasCalled);
  assert('LOCAL_ONLY + describe mode: shows a clear, honest status (real local answer, or a specific local-model message) — not a silent failure and not a stale "Local-Only" refusal',
    /local vision model/i.test(localOnlyDescribe.result.value) || (localOnlyDescribe.result.value.length>3 && !/Local-Only/i.test(localOnlyDescribe.result.value)),
    localOnlyDescribe.result.value);

  // LOCAL_ONLY: color mode DOES have a local implementation — should succeed
  // using the local read, with no cloud call.
  groqWasCalled = false;
  const localOnlyColor = await send('Runtime.evaluate', {
    expression: `(async()=>{ setMode('color'); await snap(); return document.getElementById('descEl').textContent; })()`,
    awaitPromise: true, returnByValue: true
  });
  assert('LOCAL_ONLY + color mode: does not call the cloud', !groqWasCalled, 'groqWasCalled='+groqWasCalled);
  assert('LOCAL_ONLY + color mode: still produces a usable local answer', localOnlyColor.result.value.length>3 && localOnlyColor.result.value!=='Tap the camera or press Snap.', localOnlyColor.result.value);

  // HYBRID: describe mode should call the cloud normally (existing behavior).
  groqWasCalled = false;
  await send('Runtime.evaluate', {
    expression: `(async()=>{ document.getElementById('aiModeSel').value='HYBRID'; saveAiMode(); setMode('describe'); await snap(); })()`,
    awaitPromise: true
  });
  assert('HYBRID + describe mode: calls the cloud normally', groqWasCalled, 'groqWasCalled='+groqWasCalled);

  // CLOUD_ENHANCED: color mode should skip the local shortcut and call the cloud.
  groqWasCalled = false;
  const cloudEnhancedColor = await send('Runtime.evaluate', {
    expression: `(async()=>{ document.getElementById('aiModeSel').value='CLOUD_ENHANCED'; saveAiMode(); setMode('color'); await snap(); return document.getElementById('descEl').textContent; })()`,
    awaitPromise: true, returnByValue: true
  });
  assert('CLOUD_ENHANCED + color mode: calls the cloud instead of using the local shortcut', groqWasCalled, 'groqWasCalled='+groqWasCalled);

  console.log('=== aiMode tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
