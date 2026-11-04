// Verifies Gemini's real retry-delay parsing, added after a direct check of
// this account's actual Gemini quota turned up a real surprise: the free
// tier for gemini-2.5-flash on a project without billing enabled is just 20
// requests/DAY (quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier)
// — nowhere near the ~1,500/day figure Entry 22 cited from third-party blog
// research, which was wrong for this configuration and shouldn't have been
// repeated without checking against a real response. Gemini's error does
// include a genuine retry delay in its details array (e.g. "53s"), which
// callGemini() now parses the same way Groq's wait time already is, instead
// of a vague "rate-limited too right now" message with no number.
const CDP = 'http://localhost:9333';
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

// The exact real error body captured from a live 429 response.
const REAL_GEMINI_QUOTA_ERROR = {
  error: {
    code: 429,
    message: "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 53.788831731s.",
    status: "RESOURCE_EXHAUSTED",
    details: [
      { "@type": "type.googleapis.com/google.rpc.Help", links: [{ description: "Learn more about Gemini API quotas", url: "https://ai.google.dev/gemini-api/docs/rate-limits" }] },
      { "@type": "type.googleapis.com/google.rpc.QuotaFailure", violations: [{ quotaMetric: "generativelanguage.googleapis.com/generate_content_free_tier_requests", quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier", quotaDimensions: { location: "global", model: "gemini-2.5-flash" }, quotaValue: "20" }] },
      { "@type": "type.googleapis.com/google.rpc.RetryInfo", retryDelay: "53s" }
    ]
  }
};

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
  await send('Fetch.enable', { patterns: [{ urlPattern: 'https://generativelanguage.googleapis.com/*' }] });

  ws.addEventListener('message', (msg) => {
    const data = JSON.parse(msg.data);
    if (data.method === 'Fetch.requestPaused') {
      const { requestId, request } = data.params;
      if (request.url.includes('generativelanguage.googleapis.com') && request.method === 'OPTIONS') {
        // See tests/README.md's CORS-preflight gotcha — without this, the
        // browser treats the preflight as failed and the real POST never
        // fires at all, surfacing as a generic network error instead of
        // ever reaching the mocked 429.
        send('Fetch.fulfillRequest', { requestId, responseCode: 204, responseHeaders: [
          { name: 'Access-Control-Allow-Origin', value: '*' },
          { name: 'Access-Control-Allow-Methods', value: 'POST, OPTIONS' },
          { name: 'Access-Control-Allow-Headers', value: 'Content-Type, x-goog-api-key' }
        ]});
      } else if (request.url.includes('generativelanguage.googleapis.com')) {
        const body = JSON.stringify(REAL_GEMINI_QUOTA_ERROR);
        send('Fetch.fulfillRequest', { requestId, responseCode: 429, responseHeaders: [
          { name: 'Content-Type', value: 'application/json' },
          { name: 'Access-Control-Allow-Origin', value: '*' }
        ], body: Buffer.from(body).toString('base64') });
      } else { send('Fetch.continueRequest', { requestId }); }
    }
  });

  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  const r = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        geminiKey = 'fake-gemini-key-for-this-test';
        let caught = null;
        try{ await callGemini('fakeimg', null); }catch(e){ caught = e.message; }
        geminiKey = '';
        return JSON.stringify({ caught });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d = JSON.parse(r.result.value);
  assert('Gemini\'s real "53s" retry delay is parsed into a concrete wait-time message',
    d.caught && d.caught.includes('53 seconds'), JSON.stringify(d));
  assert('the message names Gemini specifically, not a generic error',
    d.caught && /gemini/i.test(d.caught), JSON.stringify(d));

  console.log('=== Gemini quota parsing tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
