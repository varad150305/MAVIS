// Verifies two fixes made after the user reported "both Gemini and Groq keep
// rate limiting" and "it's not reading text properly":
//
// 1. Gemini's real free-tier limit for this project is just 20 requests/day
//    (found via a real 429 in callGemini() — see IMPLEMENTATION_PROGRESS.md).
//    callGroq() now tracks usage client-side (geminiBudget) and skips the
//    guaranteed-fail round trip to Gemini once today's local count says the
//    quota is gone, going straight to Groq instead.
// 2. snap() in text mode used to show nothing but a bare error when the
//    cloud call failed, even if local Tesseract OCR had already produced
//    *some* text that just didn't pass the strict single-line/high-confidence
//    gate. It now falls back to showing that local read, honestly framed as
//    possibly incomplete, instead of leaving the user with nothing.
//
// Both use direct function-stubbing (like test_provider_priority.mjs) since
// this is pure control-flow logic, not something that needs a real network
// call or a real degraded-image OCR run to verify.
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

  // ── Gemini budget indicator reflects real usage ──
  const rBudget = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        localStorage.removeItem('dk_gemini_budget');
        geminiBudget = {date:_todayStr(), used:0};
        const before = geminiBudgetExhausted();
        for(let i=0;i<20;i++) _recordGeminiUse();
        const after = geminiBudgetExhausted();
        return JSON.stringify({ before, after, used: geminiBudget.used });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const dBudget = JSON.parse(rBudget.result.value);
  assert('budget starts non-exhausted and becomes exhausted after 20 recorded uses',
    dBudget.before===false && dBudget.after===true && dBudget.used===20, JSON.stringify(dBudget));

  // ── callGroq() skips Gemini entirely once the local budget says today's quota is gone ──
  const rSkip = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        geminiKey = 'fake-gemini-key';
        geminiBudget = {date:_todayStr(), used:GEMINI_DAILY_FREE_LIMIT};
        let groqCalled=false, geminiCalled=false;
        const origGemini=callGemini, origGroqOnly=_callGroqOnly;
        window.callGemini = async()=>{ geminiCalled=true; return 'Gemini answer'; };
        window._callGroqOnly = async()=>{ groqCalled=true; return 'Groq answer'; };
        const result = await callGroq('fakeimg', null);
        window.callGemini=origGemini; window._callGroqOnly=origGroqOnly;
        geminiKey=''; geminiBudget={date:_todayStr(),used:0}; localStorage.removeItem('dk_gemini_budget');
        return JSON.stringify({ groqCalled, geminiCalled, result });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const dSkip = JSON.parse(rSkip.result.value);
  assert('once the local Gemini budget is exhausted, callGroq() skips Gemini and goes straight to Groq',
    dSkip.groqCalled && !dSkip.geminiCalled && dSkip.result==='Groq answer', JSON.stringify(dSkip));

  // ── A real Gemini per-DAY 429 marks the local budget exhausted immediately,
  // even if our own count hadn't reached 20 yet (e.g. usage from elsewhere) ──
  const rReactive = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        geminiKey='fake-gemini-key';
        geminiBudget={date:_todayStr(), used:1}; // our own count thinks there's plenty left
        const origFetch=window.fetch;
        window.fetch=async()=> new Response(JSON.stringify({error:{message:'Quota exceeded for quota metric GenerateRequestsPerDayPerProjectPerModel-FreeTier'}}), {status:429});
        let caught=null;
        try{ await callGemini('fakeimg', null); }catch(e){ caught=e.message; }
        window.fetch=origFetch;
        const nowExhausted = geminiBudgetExhausted();
        geminiKey=''; geminiBudget={date:_todayStr(),used:0}; localStorage.removeItem('dk_gemini_budget');
        return JSON.stringify({ caught, nowExhausted });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const dReactive = JSON.parse(rReactive.result.value);
  assert('a real Gemini daily-limit 429 marks the local budget exhausted immediately, and says it resets tomorrow',
    dReactive.nowExhausted===true && /free daily limit/i.test(dReactive.caught) && /resets tomorrow/i.test(dReactive.caught),
    JSON.stringify(dReactive));

  // ── Text mode: local OCR produced something (multi-line, so it doesn't
  // pass the strict single-line gate) and the cloud call then fails — the
  // local read should still reach the screen, not just an error. ──
  // startCam() alert()s and blocks forever on a fresh profile with no Groq
  // key saved (see index.html's startCam() guard) — set a fake one first,
  // matching test_auto_retry.mjs's own established pattern.
  await send('Runtime.evaluate', { expression: `localStorage.setItem('dk_groq','gsk_fake_test_key'); localStorage.setItem('dk_ai_mode','HYBRID'); location.reload();` });
  await sleep(2000);
  await send('Runtime.evaluate', { expression: `startCam()`, awaitPromise: true });
  await sleep(2500); // let startup announcements clear (matches test_auto_retry.mjs)
  const rOcrFallback = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        // Set mode directly rather than via setMode('text') — setMode()
        // auto-triggers its own real snap() when a stream already exists,
        // which would race with (and, via the snapInFlight guard, silently
        // no-op) the explicit stubbed snap() call below.
        mode = 'text';
        ocrReady = true;
        const origLocalOCR = localOCR, origCallGroq = callGroq;
        const sysMsgs=[];
        const origAddMsg = addMsg;
        window.addMsg = (role, text) => { if(role==='sys') sysMsgs.push(text); return origAddMsg(role, text); };
        window.localOCR = async()=> ({ text:'EXIT ONLY\\nDO NOT ENTER', confidence: 70, lines: 2 }); // fails the lines<=1 gate
        window.callGroq = async()=> { throw new Error('Both AI providers are unavailable right now. Gemini: rate-limited. Groq: rate-limited.'); };
        await snap();
        const textPanel = document.getElementById('textContent').textContent;
        window.localOCR = origLocalOCR; window.callGroq = origCallGroq; window.addMsg = origAddMsg;
        return JSON.stringify({ textPanel, sysMsgs });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const dOcrFallback = JSON.parse(rOcrFallback.result.value);
  assert('text mode shows the local OCR read (not a bare error) when the cloud call fails after an unconfident local read',
    /EXIT ONLY/i.test(dOcrFallback.textPanel) && /DO NOT ENTER/i.test(dOcrFallback.textPanel), JSON.stringify(dOcrFallback));
  assert('a system message explains the cloud failure alongside the local fallback, so the user knows it may be incomplete',
    dOcrFallback.sysMsgs.some(s=>/cloud reading failed/i.test(s) && /may be incomplete/i.test(s)), JSON.stringify(dOcrFallback));

  console.log('=== gemini budget + OCR fallback tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
