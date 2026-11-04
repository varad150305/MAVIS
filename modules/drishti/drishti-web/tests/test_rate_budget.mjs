// Verifies the client-side rate-limit budget tracker added after the user
// reported the rate limiting was "causing too much trouble" (see
// IMPLEMENTATION_PROGRESS.md). Instead of always firing a request and finding
// out it's rate-limited 5-15+ seconds later, the app now tracks Groq's own
// x-ratelimit-* response headers and fails fast, with a clear message, when
// it already knows the budget is exhausted.
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
  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  // ── _parseRateLimitDuration: pure logic ──
  const durTest = await send('Runtime.evaluate', {
    expression: `
      JSON.stringify({
        ms: _parseRateLimitDuration('120ms'),
        s: _parseRateLimitDuration('42.3375s'),
        m: _parseRateLimitDuration('1.5m'),
        bare: _parseRateLimitDuration('30'),
        garbage: _parseRateLimitDuration('not-a-duration'),
        empty: _parseRateLimitDuration(null)
      })
    `,
    returnByValue: true
  });
  const dt = JSON.parse(durTest.result.value);
  assert('parses "120ms" as 120', dt.ms===120, JSON.stringify(dt));
  assert('parses "42.3375s" as ~42337.5ms', Math.abs(dt.s-42337.5)<1, JSON.stringify(dt));
  assert('parses "1.5m" as 90000ms', dt.m===90000, JSON.stringify(dt));
  assert('bare number defaults to seconds', dt.bare===30000, JSON.stringify(dt));
  assert('unrecognized format returns null, not a guess', dt.garbage===null, JSON.stringify(dt));
  assert('null input returns null', dt.empty===null, JSON.stringify(dt));

  // ── Header tracking + pre-flight fast-fail (mocked fetch, no real network) ──
  const flowTest = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const orig = window.fetch;
        let realCallCount = 0;
        window.fetch = async (url, opts) => {
          if (typeof url==='string' && url.includes('api.groq.com')) {
            realCallCount++;
            return new Response(JSON.stringify({choices:[{message:{content:'A chair.'},finish_reason:'stop'}]}), {
              status: 200,
              headers: { 'x-ratelimit-remaining-tokens': '150', 'x-ratelimit-reset-tokens': '20s', 'Access-Control-Allow-Origin': '*' }
            });
          }
          return orig(url, opts);
        };
        aiMode = 'HYBRID';
        const before = JSON.stringify(groqBudget);
        await callGroq(lastImg||'x', null); // real call — response says only 150 tokens left
        const afterFirstCall = JSON.stringify({...groqBudget});
        let secondCallThrew = null;
        try { await callGroq(lastImg||'x', null); } catch(e) { secondCallThrew = e.message; } // should fail FAST, not hit the network
        window.fetch = orig;
        return JSON.stringify({ before, afterFirstCall, realCallCount, secondCallThrew });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const ft = JSON.parse(flowTest.result.value);
  const afterFirst = JSON.parse(ft.afterFirstCall);
  assert('budget starts unknown (null) before any real response', JSON.parse(ft.before).remainingTokens===null, ft.before);
  assert('budget updates from the response headers after a real call', afterFirst.remainingTokens===150, ft.afterFirstCall);
  assert('a second call, with budget known-low, does NOT hit the network again', ft.realCallCount===1, JSON.stringify(ft));
  assert('the second call fails fast with a clear, specific message', ft.secondCallThrew && ft.secondCallThrew.includes('quota is low') && ft.secondCallThrew.includes('seconds'), JSON.stringify(ft));

  // ── UI indicator reflects the tracked state ──
  const uiTest = await send('Runtime.evaluate', {
    expression: `
      (function(){
        groqBudget = {remainingTokens: 150, resetAt: Date.now()+20000};
        updateGroqBudgetIndicator();
        const pill = document.getElementById('budgetPill');
        return JSON.stringify({visible: pill.style.display!=='none', className: pill.className, text: pill.textContent});
      })()
    `,
    returnByValue: true
  });
  const ui = JSON.parse(uiTest.result.value);
  assert('low budget shows the pill with the "low" style', ui.visible && ui.className.includes('low'), JSON.stringify(ui));
  assert('low budget pill shows the actual remaining count', ui.text.includes('150'), JSON.stringify(ui));

  console.log('=== rate budget tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
