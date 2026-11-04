// Verifies the auto-retry-on-rate-limit feature added after the user asked
// "how can we tackle this rate limit thing" — instead of just failing fast
// with a wait-time message (Entry 15) and making the user notice and manually
// retry, snap() now counts the wait down visibly and retries automatically
// once. Uses CDP Fetch interception to serve a real 429 (with Groq's actual
// error message shape) on the first request and a real success on the
// second, so this exercises the actual retry path end-to-end, not a mock of
// the retry logic itself.
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

  let groqCallCount = 0;
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
        groqCallCount++;
        if (groqCallCount === 1) {
          // Real Groq 429 error shape (see IMPLEMENTATION_PROGRESS.md Entry 10/11).
          const body = JSON.stringify({ error: { message: 'Rate limit reached. Please try again in 2.5s.' } });
          send('Fetch.fulfillRequest', { requestId, responseCode: 429, responseHeaders: [
            { name: 'Content-Type', value: 'application/json' },
            { name: 'Access-Control-Allow-Origin', value: '*' }
          ], body: Buffer.from(body).toString('base64') });
        } else {
          const body = JSON.stringify({ choices: [{ message: { content: 'A wooden chair by the window.' }, finish_reason: 'stop' }] });
          send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [
            { name: 'Content-Type', value: 'application/json' },
            { name: 'Access-Control-Allow-Origin', value: '*' }
          ], body: Buffer.from(body).toString('base64') });
        }
      } else { send('Fetch.continueRequest', { requestId }); }
    }
  });

  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);
  await send('Runtime.evaluate', { expression: `localStorage.setItem('dk_groq','gsk_fake_test_key'); localStorage.setItem('dk_ai_mode','HYBRID'); location.reload();` });
  await sleep(2000);
  await send('Runtime.evaluate', { expression: `startCam()`, awaitPromise: true });
  await sleep(2500); // let the startup announcements clear

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  const r = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const statusLog=[];
        const origSetStatus = setStatus;
        window.setStatus = (cls, text) => { statusLog.push(text); origSetStatus(cls, text); };
        setMode('describe');
        await snap(); // first attempt -> 429 -> should schedule a 3s auto-retry (ceil(2.5)=3)
        const scheduledRightAfterFailure = !!pendingRetryTimer;
        // Wait past the retry delay for the automatic second attempt to fire and complete.
        await new Promise(res => setTimeout(res, 4500));
        window.setStatus = origSetStatus;
        return JSON.stringify({ scheduledRightAfterFailure, finalDesc: document.getElementById('descEl').textContent });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d = JSON.parse(r.result.value);
  assert('a retry was actually scheduled right after the first failure',
    d.scheduledRightAfterFailure, JSON.stringify(d));
  assert('exactly 2 real requests were made — the failed attempt and the one automatic retry',
    groqCallCount === 2, 'groqCallCount='+groqCallCount);
  assert('the automatic retry succeeded and the real result made it to the screen',
    d.finalDesc.includes('wooden chair'), JSON.stringify(d));

  // A retry attempt that itself fails again must NOT schedule a second
  // auto-retry — otherwise a genuinely stuck-limited account gets hammered
  // forever, exactly the bug Guardian Mode had (Entry 12). Tested directly
  // against the real snap(isRetry) flag rather than re-choreographing the
  // network mock a second time.
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        _cancelPendingRetry();
        const origCallGroq = callGroq;
        window.callGroq = async () => { throw new Error('Groq quota is low right now — please wait about 5 seconds before trying again.'); };
        await snap(true); // simulate this being the automatic retry itself
        const scheduledAnotherRetry = !!pendingRetryTimer;
        window.callGroq = origCallGroq;
        _cancelPendingRetry();
        return JSON.stringify({ scheduledAnotherRetry });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('a retry that fails again does not chain into another auto-retry',
    d2.scheduledAnotherRetry === false, JSON.stringify(d2));

  console.log('=== auto-retry tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
