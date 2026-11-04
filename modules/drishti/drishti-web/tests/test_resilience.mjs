// Verifies three resilience fixes added after real-key testing surfaced real
// problems (see IMPLEMENTATION_PROGRESS.md):
// 1. Truncation retry — callGroq() automatically retries with a larger token
//    budget when a reasoning model gets cut off before producing an answer,
//    instead of permanently guessing one fixed budget.
// 2. Adaptive performance scaling — the on-device object detector measures its
//    own real inference latency and downgrades to a lighter model if it's
//    running too slowly to be useful in real time.
// 3. Rate-limit messaging — a 429 error surfaces Groq's actual wait time
//    instead of a vague "try again in a moment", verified against the real
//    error message format returned by this account during testing.
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

  // Groq mock: first call for a given "session" returns a truncated
  // (finish_reason 'length', empty content) reasoning-only response; the
  // second call (the retry) returns a real answer. Lets us verify callGroq()
  // actually retries with a bigger budget rather than giving up immediately.
  let groqCallCount = 0;
  let capturedMaxTokens = [];
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
        let reqBody = {};
        try { reqBody = JSON.parse(request.postData || '{}'); } catch(e){}
        capturedMaxTokens.push(reqBody.max_tokens);
        const truncated = groqCallCount === 1;
        const respBody = JSON.stringify({
          choices: [{
            message: { content: truncated ? '' : 'A chair is directly ahead.' },
            finish_reason: truncated ? 'length' : 'stop'
          }]
        });
        send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [
          { name: 'Content-Type', value: 'application/json' },
          { name: 'Access-Control-Allow-Origin', value: '*' }
        ], body: Buffer.from(respBody).toString('base64') });
      } else { send('Fetch.continueRequest', { requestId }); }
    }
  });

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Network.enable');
  await send('Fetch.enable', { patterns: [{ urlPattern: 'https://api.groq.com/*' }] });
  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);
  await send('Runtime.evaluate', { expression: `localStorage.setItem('dk_groq','gsk_fake_test_key'); localStorage.setItem('dk_ai_mode','HYBRID'); location.reload();` });
  await sleep(2000);
  await send('Runtime.evaluate', { expression: `startCam()`, awaitPromise: true });
  await sleep(1000);

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  // ── Truncation retry ──
  const r = await send('Runtime.evaluate', {
    expression: `(async()=>{ try { const r = await callGroq(lastImg||'', null); return JSON.stringify({ok:true, r}); } catch(e){ return JSON.stringify({ok:false, e:e.message}); } })()`,
    awaitPromise: true, returnByValue: true
  });
  const parsed = JSON.parse(r.result.value);
  assert('truncation retry: callGroq makes exactly 2 requests (initial + retry)', groqCallCount===2, 'groqCallCount='+groqCallCount);
  assert('truncation retry: second request uses a larger max_tokens than the first', capturedMaxTokens[1] > capturedMaxTokens[0], JSON.stringify(capturedMaxTokens));
  assert('truncation retry: returns the real answer from the successful retry, not "incomplete"', parsed.ok && parsed.r==='A chair is directly ahead.', JSON.stringify(parsed));

  // ── Adaptive performance scaling ──
  const odResult = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        // Simulate a model that's genuinely too slow, without needing 4+ real
        // seconds per call or a real TF.js load — a stub with the same shape
        // odLoop actually calls (odModel.detect), timed via a real setTimeout
        // so performance.now() sees real elapsed time, not a mocked clock.
        odModel = { detect: () => new Promise(res => setTimeout(() => res([]), 900)) };
        odReady = true; odOn = true; odDowngraded = false; odInferenceTimes = [];
        let downgradeCalled = false;
        const _orig = downgradeObjectDetector;
        downgradeObjectDetector = async () => { downgradeCalled = true; odDowngraded = true; };
        // Run the loop body directly 3 times (enough to fill the measurement window)
        for (let i=0; i<3; i++) {
          const t0 = performance.now();
          await odModel.detect();
          const dt = performance.now()-t0;
          odInferenceTimes.push(dt);
          if (odInferenceTimes.length>5) odInferenceTimes.shift();
          if (!odDowngraded && odInferenceTimes.length>=3) {
            const avg = odInferenceTimes.reduce((a,b)=>a+b,0)/odInferenceTimes.length;
            if (avg > OD_SLOW_THRESHOLD_MS) await downgradeObjectDetector();
          }
        }
        downgradeObjectDetector = _orig;
        odOn = false;
        return JSON.stringify({downgradeCalled, avgMs: odInferenceTimes.reduce((a,b)=>a+b,0)/odInferenceTimes.length});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const odParsed = JSON.parse(odResult.result.value);
  assert('adaptive scaling: sustained slow inference (900ms avg) triggers a downgrade', odParsed.downgradeCalled, JSON.stringify(odParsed));

  // Fast model should NOT trigger a downgrade
  const odFastResult = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        odModel = { detect: () => new Promise(res => setTimeout(() => res([]), 50)) };
        odDowngraded = false; odInferenceTimes = [];
        let downgradeCalled = false;
        const _orig = downgradeObjectDetector;
        downgradeObjectDetector = async () => { downgradeCalled = true; };
        for (let i=0; i<3; i++) {
          const t0 = performance.now();
          await odModel.detect();
          const dt = performance.now()-t0;
          odInferenceTimes.push(dt);
          if (odInferenceTimes.length>5) odInferenceTimes.shift();
          if (!odDowngraded && odInferenceTimes.length>=3) {
            const avg = odInferenceTimes.reduce((a,b)=>a+b,0)/odInferenceTimes.length;
            if (avg > OD_SLOW_THRESHOLD_MS) await downgradeObjectDetector();
          }
        }
        downgradeObjectDetector = _orig;
        return JSON.stringify({downgradeCalled});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  assert('adaptive scaling: fast inference (50ms avg) does not trigger a downgrade', !JSON.parse(odFastResult.result.value).downgradeCalled);

  // downgradeObjectDetector itself is idempotent (won't double-fire even if called twice)
  const idempotentResult = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        odDowngraded = true; // simulate already-downgraded state
        let loadCalled = false;
        window.cocoSsd = { load: async () => { loadCalled = true; return {}; } };
        await downgradeObjectDetector();
        return JSON.stringify({loadCalled});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  assert('downgradeObjectDetector is a no-op if already downgraded', !JSON.parse(idempotentResult.result.value).loadCalled);

  // ── Rate-limit messaging ──
  // Reconfigure the mock: OPTIONS still succeeds, but the POST now returns a
  // 429 with the exact error message shape Groq actually returned during
  // testing against the real account (verbatim structure, made-up numbers).
  const r429 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const orig = window.fetch;
        window.fetch = async (url, opts) => {
          if (typeof url==='string' && url.includes('api.groq.com') && opts && opts.method==='POST') {
            return new Response(JSON.stringify({error:{message:"Rate limit reached for model \`qwen/qwen3.6-27b\` in organization \`org_test\` service tier \`on_demand\` on tokens per minute (TPM): Limit 8000, Used 6917, Requested 6728. Please try again in 42.3375s. Need more tokens? Upgrade to Dev Tier today."}}), {status:429});
          }
          return orig(url, opts);
        };
        try {
          await callGroq(lastImg||'x', null);
          window.fetch = orig;
          return JSON.stringify({threw:false});
        } catch(e) {
          window.fetch = orig;
          return JSON.stringify({threw:true, message:e.message});
        }
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const r429Parsed = JSON.parse(r429.result.value);
  assert('rate limit: throws an error (not a silent failure)', r429Parsed.threw, JSON.stringify(r429Parsed));
  assert('rate limit: extracts the real wait time from Groq\'s message ("about 43 seconds")', r429Parsed.message && r429Parsed.message.includes('43 seconds'), JSON.stringify(r429Parsed));

  // ── Retry-failure error consistency ──
  // Real bug found via live testing: when the truncation-retry itself failed
  // (e.g. hit a 429), the error thrown was a bare "Groq error 429" instead of
  // the same helpful, wait-time-parsed message the primary path produces.
  const retryFailResult = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const orig = window.fetch;
        let call = 0;
        window.fetch = async (url, opts) => {
          if (typeof url==='string' && url.includes('api.groq.com') && opts && opts.method==='POST') {
            call++;
            if (call===1) {
              // First call: truncated, triggers the retry
              return new Response(JSON.stringify({choices:[{message:{content:''},finish_reason:'length'}]}), {status:200});
            }
            // Retry call: fails with a real Groq rate-limit body
            return new Response(JSON.stringify({error:{message:"Rate limit reached for model \`qwen/qwen3.6-27b\`... Please try again in 8.5s. ..."}}), {status:429});
          }
          return orig(url, opts);
        };
        try {
          await callGroq(lastImg||'x', null);
          window.fetch = orig;
          return JSON.stringify({threw:false});
        } catch(e) {
          window.fetch = orig;
          return JSON.stringify({threw:true, message:e.message, callCount:call});
        }
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const retryFail = JSON.parse(retryFailResult.result.value);
  assert('retry-failure: makes exactly 2 requests (initial truncation + retry)', retryFail.callCount===2, JSON.stringify(retryFail));
  assert('retry-failure: surfaces the parsed rate-limit wait time, not a bare "Groq error 429"', retryFail.message && retryFail.message.includes('9 seconds') && !retryFail.message.includes('Groq error 429'), JSON.stringify(retryFail));

  // ── Guardian cooldown paces every attempt, not just alerts ──
  // Real bug found via live testing: lastGuardianAlert only updated on an
  // actual DANGER/CAUTION rating, so a persistent hazard rated SAFE (or an
  // errored call) never reset the cooldown — handleGuardianProximity fired a
  // fresh real API call on almost every detection tick, hammering the account's
  // rate limit continuously.
  const guardianCooldownResult = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const orig = window.fetch;
        let realCallCount = 0;
        window.fetch = async (url, opts) => {
          if (typeof url==='string' && url.includes('api.groq.com') && opts && opts.method==='POST') {
            realCallCount++;
            return new Response(JSON.stringify({choices:[{message:{content:'SAFE\\nAll clear.'}, finish_reason:'stop'}]}), {status:200});
          }
          if (typeof url==='string' && url.includes('api.groq.com') && opts && opts.method==='OPTIONS') {
            return new Response(null, {status:204, headers:{'Access-Control-Allow-Origin':'*','Access-Control-Allow-Methods':'POST, OPTIONS','Access-Control-Allow-Headers':'Content-Type'}});
          }
          return orig(url, opts);
        };
        lastGuardianAttempt = 0; lastGuardianAlert = 0; guardianOn = true;
        // Simulate a persistent close hazard triggering handleGuardianProximity
        // on 5 consecutive "detection ticks" in immediate succession — exactly
        // what happens in odLoop when a hazard stays in frame across ticks.
        const fakePreds = [{class:'person', score:0.9, bbox:[0,0, odCanvas.width*0.5, odCanvas.height*0.5]}];
        for (let i=0;i<5;i++){
          handleGuardianProximity(fakePreds);
          await new Promise(r=>setTimeout(r,50)); // ticks are ~300-500ms apart in reality; even this short gap should still be within the cooldown
        }
        await new Promise(r=>setTimeout(r,300)); // let any in-flight call settle
        guardianOn = false;
        window.fetch = orig;
        return JSON.stringify({realCallCount});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const gc = JSON.parse(guardianCooldownResult.result.value);
  assert('guardian cooldown: a SAFE (non-alert) result still paces subsequent attempts — at most 1 real call across 5 rapid ticks', gc.realCallCount<=1, JSON.stringify(gc));

  console.log('=== resilience tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
