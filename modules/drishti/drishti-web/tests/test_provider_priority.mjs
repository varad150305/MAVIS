// Verifies the provider-priority flip from Entry 26: when a Gemini key is
// configured, callGroq() (the single entry point every caller uses) now
// tries Gemini FIRST, with Groq as the fallback if Gemini fails — the
// reverse of the original Entry 23 arrangement. Uses mocked fetch for BOTH
// providers (no real API keys/costs) — the real-key end-to-end proof that
// each provider's request/response shape actually works already exists from
// Entries 23-25's live verification; this file is about the ORDER and
// fallback behavior, which is pure logic and doesn't need real keys to test.
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

  // ── With a Gemini key configured: Gemini goes first, Groq is never touched ──
  const r1 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        geminiKey = 'fake-gemini-key';
        let groqCalled = false, geminiCalled = false;
        const origGemini = callGemini;
        const origGroqOnly = _callGroqOnly;
        window.callGemini = async (...args) => { geminiCalled = true; return 'A Gemini answer.'; };
        window._callGroqOnly = async (...args) => { groqCalled = true; return 'A Groq answer.'; };
        const result = await callGroq('fakeimg', null);
        window.callGemini = origGemini;
        window._callGroqOnly = origGroqOnly;
        geminiKey = '';
        return JSON.stringify({ groqCalled, geminiCalled, result });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d1 = JSON.parse(r1.result.value);
  assert('with a Gemini key set, Gemini is tried first and Groq is never called',
    d1.geminiCalled && !d1.groqCalled && d1.result==='A Gemini answer.', JSON.stringify(d1));

  // ── If Gemini fails, Groq is used as the fallback ──
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        geminiKey = 'fake-gemini-key';
        const sysMsgs=[];
        const origAddMsg = addMsg;
        window.addMsg = (role, text) => { if(role==='sys') sysMsgs.push(text); return origAddMsg(role, text); };
        const origGemini = callGemini;
        const origGroqOnly = _callGroqOnly;
        window.callGemini = async () => { throw new Error('Gemini is down.'); };
        window._callGroqOnly = async () => 'A Groq fallback answer.';
        const result = await callGroq('fakeimg', null);
        window.callGemini = origGemini;
        window._callGroqOnly = origGroqOnly;
        window.addMsg = origAddMsg;
        geminiKey = '';
        return JSON.stringify({ result, sysMsgs });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('if Gemini fails, Groq is used as the fallback and a system message explains it',
    d2.result==='A Groq fallback answer.' && d2.sysMsgs.some(s=>/groq fallback/i.test(s)), JSON.stringify(d2));

  // ── If BOTH fail, the message must say so honestly, not just show Gemini's
  // error alone — found from real usage: the user saw a Gemini-only rate-
  // limit message and (reasonably) couldn't tell Groq had been tried and
  // failed too, since nothing said so. ──
  const r3 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        geminiKey = 'fake-gemini-key';
        const origGemini = callGemini;
        const origGroqOnly = _callGroqOnly;
        window.callGemini = async () => { throw new Error('Gemini error message.'); };
        window._callGroqOnly = async () => { throw new Error('Groq error message.'); };
        let caught = null;
        try{ await callGroq('fakeimg', null); }catch(e){ caught = e.message; }
        window.callGemini = origGemini;
        window._callGroqOnly = origGroqOnly;
        geminiKey = '';
        return JSON.stringify({ caught });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d3 = JSON.parse(r3.result.value);
  assert('if both fail, the message mentions both providers, not just Gemini alone',
    /gemini/i.test(d3.caught) && /groq/i.test(d3.caught) && /groq error message/i.test(d3.caught),
    JSON.stringify(d3));

  // ── With NO Gemini key: behavior is exactly the pre-Entry-23 original — Groq only ──
  const r4 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        // geminiKey is '' by default on a fresh load / after the resets above.
        let groqCalled = false, geminiCalled = false;
        const origGemini = callGemini;
        const origGroqOnly = _callGroqOnly;
        window.callGemini = async () => { geminiCalled = true; return 'A Gemini answer.'; };
        window._callGroqOnly = async () => { groqCalled = true; return 'A Groq answer.'; };
        const result = await callGroq('fakeimg', null);
        window.callGemini = origGemini;
        window._callGroqOnly = origGroqOnly;
        return JSON.stringify({ groqCalled, geminiCalled, result, geminiKeyNow: geminiKey });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d4 = JSON.parse(r4.result.value);
  assert('with no Gemini key configured, only Groq is ever called (unchanged original behavior)',
    d4.groqCalled && !d4.geminiCalled && d4.result==='A Groq answer.', JSON.stringify(d4));

  console.log('=== provider priority tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
