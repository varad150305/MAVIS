// Reproduces, verbatim, the exact real error the user reported: "Request too
// large for model qwen/qwen3.6-27b ... Limit 8000, Requested 9805, please
// reduce your message size and try again." Two real bugs, both fixed here:
// (1) retrying the identical request — even after waiting — can NEVER
// succeed, since the request itself (prompt + requested max_tokens) already
// exceeds the account's entire per-minute ceiling; the old code treated this
// as a normal rate limit and told the user to "wait a moment," which is
// actively wrong advice for this specific failure. (2) Groq's own error
// message gives us the exact numbers needed to compute a smaller max_tokens
// that fits — _requestWithSizeGuard() now uses them to auto-retry once with
// a corrected budget instead of failing outright.
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
  await send('Fetch.enable', { patterns: [{ urlPattern: 'https://api.groq.com/*' }] });

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  const REAL_TOO_LARGE_MSG = 'Request too large for model `qwen/qwen3.6-27b` in organization `org_01kw927bnyfqya8mg2wy6gt10x` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 9805, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing';

  // ── Scenario A: the retry (with a corrected, smaller max_tokens) succeeds ──
  let callNum = 0;
  let secondCallMaxTokens = null;
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
        callNum++;
        if (callNum === 1) {
          // First attempt uses DEFAULT_MAX_TOKENS=3000 — real error, real numbers.
          const body = JSON.stringify({ error: { message: REAL_TOO_LARGE_MSG } });
          send('Fetch.fulfillRequest', { requestId, responseCode: 429, responseHeaders: [
            { name: 'Content-Type', value: 'application/json' },
            { name: 'Access-Control-Allow-Origin', value: '*' }
          ], body: Buffer.from(body).toString('base64') });
        } else {
          const reqBody = JSON.parse(Buffer.from(request.postData || '', 'utf8').toString());
          secondCallMaxTokens = reqBody.max_tokens;
          const body = JSON.stringify({ choices: [{ message: { content: 'A recovered answer.' }, finish_reason: 'stop' }] });
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
  await sleep(2500);

  const r = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const sysMsgs=[];
        const origAddMsg = addMsg;
        window.addMsg = (role, text) => { if(role==='sys') sysMsgs.push(text); return origAddMsg(role, text); };
        setMode('describe');
        await snap();
        window.addMsg = origAddMsg;
        return JSON.stringify({ sysMsgs, finalDesc: document.getElementById('descEl').textContent });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d = JSON.parse(r.result.value);
  assert('exactly 2 requests were made — the failed too-large attempt and one corrected retry',
    callNum === 2, 'callNum='+callNum);
  assert('a "too large" system message was shown explaining the retry, not silence',
    d.sysMsgs.some(s=>/too large/i.test(s)), JSON.stringify(d));
  assert('the corrected retry succeeded and the real answer reached the screen',
    d.finalDesc.includes('recovered answer'), JSON.stringify(d));
  // Requested 9805 - maxTokens 3000 = 6805 real prompt tokens; safe budget =
  // 8000 - 6805 - 100 margin = 1095. Computed from Groq's own numbers, not guessed.
  assert('the retry used the precisely-computed safe max_tokens (1095), not a guess',
    secondCallMaxTokens === 1095, 'secondCallMaxTokens='+secondCallMaxTokens);

  // ── Scenario B: the error message itself, when it can't be auto-fixed,
  // must NOT say "wait a moment" — that's actively wrong advice for this
  // failure (waiting never helps a too-large request). Tested directly
  // against _throwForFailedResponse with a synthetic Response-like object,
  // since forcing the size-guard itself to fail would need a contradictory
  // mock (large enough to not be trimmable, which the guard already checks
  // via safeMax>=200) — simpler and just as valid to check the message
  // function's own output directly.
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const fakeRes = { status: 429 };
        try{
          await _throwForFailedResponse(fakeRes, undefined, ${JSON.stringify(REAL_TOO_LARGE_MSG)});
          return JSON.stringify({threw:false});
        }catch(e){
          return JSON.stringify({threw:true, message:e.message});
        }
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('the too-large error message never tells the user to "wait a moment" (waiting never helps this failure)',
    d2.threw && !/wait a moment/i.test(d2.message), JSON.stringify(d2));
  assert('the too-large error message explains the real cause instead',
    d2.threw && /too large/i.test(d2.message), JSON.stringify(d2));

  console.log('=== too-large-request tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
