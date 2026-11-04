// Verifies Cloudflare Workers AI as a third fallback provider, added after
// the user asked "what are some other things we can do apart from groq and
// gemini" once Entry 27 established both original providers can be
// genuinely, simultaneously exhausted on a real day. callGroq()'s wrapper
// now tries Gemini, then Groq, then Cloudflare (only if configured) — this
// file is pure control-flow logic (mirrors test_provider_priority.mjs's
// function-stubbing approach), NOT a real-credentials test: this session has
// no real Cloudflare account/token, so callCloudflare()'s actual HTTP
// request/response shape against the live API remains unverified (see the
// honesty note in callCloudflare()'s own comment in index.html).
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

  // ── Cloudflare is only ever tried after BOTH Gemini and Groq fail ──
  const r1 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        geminiKey='fake-gemini-key'; cfAccountId='fake-acc'; cfToken='fake-tok';
        const origGemini=callGemini, origGroqOnly=_callGroqOnly, origCf=callCloudflare;
        let geminiCalled=false, groqCalled=false, cfCalled=false;
        window.callGemini=async()=>{ geminiCalled=true; throw new Error('Gemini down.'); };
        window._callGroqOnly=async()=>{ groqCalled=true; throw new Error('Groq down.'); };
        window.callCloudflare=async()=>{ cfCalled=true; return 'A Cloudflare answer.'; };
        const sysMsgs=[]; const origAddMsg=addMsg;
        window.addMsg=(role,text)=>{ if(role==='sys') sysMsgs.push(text); return origAddMsg(role,text); };
        const result = await callGroq('fakeimg', null);
        window.callGemini=origGemini; window._callGroqOnly=origGroqOnly; window.callCloudflare=origCf; window.addMsg=origAddMsg;
        geminiKey=''; cfAccountId=''; cfToken='';
        return JSON.stringify({ geminiCalled, groqCalled, cfCalled, result, sysMsgs });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d1 = JSON.parse(r1.result.value);
  assert('Cloudflare is only tried after both Gemini and Groq have failed, and its real answer reaches the caller',
    d1.geminiCalled && d1.groqCalled && d1.cfCalled && d1.result==='A Cloudflare answer.', JSON.stringify(d1));
  assert('a system message explains the Cloudflare fallback fired',
    d1.sysMsgs.some(s=>/cloudflare/i.test(s)), JSON.stringify(d1));

  // ── If ALL THREE fail, the combined message names all three real errors ──
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        geminiKey='fake-gemini-key'; cfAccountId='fake-acc'; cfToken='fake-tok';
        const origGemini=callGemini, origGroqOnly=_callGroqOnly, origCf=callCloudflare;
        window.callGemini=async()=>{ throw new Error('Gemini msg.'); };
        window._callGroqOnly=async()=>{ throw new Error('Groq msg.'); };
        window.callCloudflare=async()=>{ throw new Error('Cloudflare msg.'); };
        let caught=null;
        try{ await callGroq('fakeimg', null); }catch(e){ caught=e.message; }
        window.callGemini=origGemini; window._callGroqOnly=origGroqOnly; window.callCloudflare=origCf;
        geminiKey=''; cfAccountId=''; cfToken='';
        return JSON.stringify({ caught });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('if all three fail, the combined message names all three providers and their real errors verbatim',
    /gemini msg\./i.test(d2.caught) && /groq msg\./i.test(d2.caught) && /cloudflare msg\./i.test(d2.caught), JSON.stringify(d2));

  // ── With NO Cloudflare configured: behavior is unchanged from before this feature existed ──
  const r3 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        aiMode='HYBRID';
        // No geminiKey, no cfAccountId/cfToken — the original single-provider path.
        const origGroqOnly=_callGroqOnly, origCf=callCloudflare;
        let cfCalled=false;
        window._callGroqOnly=async()=>{ throw new Error('Groq alone failed.'); };
        window.callCloudflare=async()=>{ cfCalled=true; return 'should not be reached'; };
        let caught=null;
        try{ await callGroq('fakeimg', null); }catch(e){ caught=e.message; }
        window._callGroqOnly=origGroqOnly; window.callCloudflare=origCf;
        return JSON.stringify({ caught, cfCalled });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d3 = JSON.parse(r3.result.value);
  assert('with no Cloudflare configured, Cloudflare is never called and the lone Groq error is re-thrown completely unwrapped',
    d3.cfCalled===false && d3.caught==='Groq alone failed.', JSON.stringify(d3));

  console.log('=== Cloudflare fallback tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
