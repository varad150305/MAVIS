// Found live: Groq's per-DAY token limit (separate from the per-MINUTE limit
// this app was originally tuned around) reports its wait time in a different
// format — "18m50.976s" instead of the plain "42.3375s" the per-minute limit
// uses. The old regex (/try again in ([\d.]+)s/i) only matched the plain
// form, so a daily-limit wait silently fell through to the vague "please
// wait a moment" message — actively unhelpful for a ~19-minute wait. This
// was discovered because a real Groq account, used throughout this session,
// hit it for real (197,112 of 200,000 daily tokens used) — not a
// hypothetical edge case.
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

  // The exact real message this app received live from a real Groq account.
  const REAL_TPD_MSG = 'Rate limit reached for model `qwen/qwen3.6-27b` in organization `org_01kw927bnyfqya8mg2wy6gt10x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 197112, Requested 5506. Please try again in 18m50.976s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing';

  const r = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const fakeRes = { status: 429 };
        try{
          await _throwForFailedResponse(fakeRes, undefined, ${JSON.stringify(REAL_TPD_MSG)});
          return JSON.stringify({threw:false});
        }catch(e){
          return JSON.stringify({threw:true, message:e.message});
        }
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d = JSON.parse(r.result.value);
  // 18m50.976s = 18*60 + 50.976 = 1130.976s, ceil = 1131.
  assert('the "18m50.976s" daily-limit format is parsed correctly (1131 seconds), not left as a vague "wait a moment"',
    d.threw && d.message.includes('1131 seconds'), JSON.stringify(d));
  assert('this is NOT misclassified as the unfixable "too large" case — it genuinely resolves after waiting',
    d.threw && !/waiting won.t help/i.test(d.message), JSON.stringify(d));

  // The plain-seconds format (per-minute limit) must still work exactly as before.
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        try{
          await _throwForFailedResponse({status:429}, undefined, 'Rate limit reached. Please try again in 42.3375s.');
          return JSON.stringify({threw:false});
        }catch(e){ return JSON.stringify({threw:true, message:e.message}); }
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('the plain-seconds per-minute format still parses correctly (unaffected by the fix)',
    d2.threw && d2.message.includes('43 seconds'), JSON.stringify(d2));

  console.log('=== daily-limit parsing tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
