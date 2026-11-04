// Verifies the Crossing-mode safety redesign (see ARCHITECTURE_AUDIT.md §7):
// the app must never speak or display a GO/WAIT-style crossing verdict, must
// never speak the raw CLEAR_VIEW/UNCLEAR_VIEW marker token, and must always
// include the fixed, code-generated disclaimer in both the spoken text and the
// visual badge — regardless of what the model returns.
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
      const h=(ev)=>{const d=JSON.parse(ev.data); if(d.id===myId){ws.removeEventListener('message',h); resolve(d.error?{error:d.error}:d.result);}};
      ws.addEventListener('message',h);
      ws.send(JSON.stringify({id:myId,method,params}));
    });
  }

  let fulfillText = 'CLEAR_VIEW\nA car is approaching from your 3 o\'clock, about 10 meters away. No pedestrian signal is visible.';

  ws.addEventListener('message', (msg) => {
    const data = JSON.parse(msg.data);
    // See test_full_regression_sweep.mjs for why the OPTIONS preflight must be
    // handled separately with real CORS headers, or the real POST never fires.
    if (data.method === 'Fetch.requestPaused') {
      const { requestId, request } = data.params;
      if (request.url.includes('api.groq.com') && request.method === 'OPTIONS') {
        send('Fetch.fulfillRequest', {
          requestId, responseCode: 204,
          responseHeaders: [
            { name: 'Access-Control-Allow-Origin', value: '*' },
            { name: 'Access-Control-Allow-Methods', value: 'POST, OPTIONS' },
            { name: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' }
          ]
        });
      } else if (request.url.includes('api.groq.com')) {
        const body = JSON.stringify({ choices: [{ message: { content: fulfillText } }] });
        send('Fetch.fulfillRequest', {
          requestId, responseCode: 200,
          responseHeaders: [
            { name: 'Content-Type', value: 'application/json' },
            { name: 'Access-Control-Allow-Origin', value: '*' }
          ],
          body: Buffer.from(body).toString('base64')
        });
      } else {
        send('Fetch.continueRequest', { requestId });
      }
    }
  });

  await send('Runtime.enable');
  await send('Network.enable');
  await send('Page.enable');
  await send('Fetch.enable', { patterns: [{ urlPattern: 'https://api.groq.com/*' }] });
  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);

  await send('Runtime.evaluate', { expression: `localStorage.setItem('dk_groq','gsk_fake_test_key'); localStorage.setItem('dk_ai_mode','HYBRID'); location.reload();` });
  await sleep(2000);
  await send('Runtime.evaluate', { expression: `startCam()`, awaitPromise: true });
  await sleep(1000);

  async function runScenario(name, groqText){
    fulfillText = groqText;
    const r = await send('Runtime.evaluate', {
      expression: `
        (async()=>{
          window.__spokenTexts = window.__spokenTexts || [];
          if(!window.__origSpeak) window.__origSpeak = speak;
          speak = function(t){ window.__spokenTexts.push(t); };
          setMode('crossing');
          await snap();
          const badge = document.getElementById('safetyBadge');
          const level = document.getElementById('safetyLevel').textContent;
          const detail = document.getElementById('safetyDetail').textContent;
          const lastSpoken = window.__spokenTexts[window.__spokenTexts.length-1] || '';
          speak = window.__origSpeak;
          return JSON.stringify({badgeClass: badge.className, level, detail, lastSpoken});
        })()
      `,
      awaitPromise: true, returnByValue: true
    });
    return JSON.parse(r.result.value);
  }

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  const clear = await runScenario('clear', "CLEAR_VIEW\nA car is approaching from your 3 o'clock, about 10 meters away. No pedestrian signal is visible.");
  assert('CLEAR_VIEW: badge never shows a GO/SAFE-style verdict', !/\bGO\b/.test(clear.level) && !clear.badgeClass.includes(' SAFE'), JSON.stringify(clear));
  assert('CLEAR_VIEW: badge label signals user judgment, not a verdict', clear.level.includes('YOUR CALL'), JSON.stringify(clear));
  assert('CLEAR_VIEW: disclaimer present in visual detail text', clear.detail.includes('does not decide for you'), JSON.stringify(clear));
  assert('CLEAR_VIEW: spoken text does not contain the raw marker token', !clear.lastSpoken.includes('CLEAR_VIEW'), JSON.stringify(clear));
  assert('CLEAR_VIEW: spoken text includes the disclaimer', clear.lastSpoken.includes('does not decide for you'), JSON.stringify(clear));
  assert('CLEAR_VIEW: spoken text includes the actual observation', clear.lastSpoken.includes('3 o\'clock'), JSON.stringify(clear));

  const unclear = await runScenario('unclear', "UNCLEAR_VIEW\nThe view is partially obstructed and the signal cannot be seen clearly.");
  assert('UNCLEAR_VIEW: badge never shows a WAIT/DANGER-as-verdict label', !/\bWAIT\b/.test(unclear.level), JSON.stringify(unclear));
  assert('UNCLEAR_VIEW: spoken text does not contain the raw marker token', !unclear.lastSpoken.includes('UNCLEAR_VIEW'), JSON.stringify(unclear));
  assert('UNCLEAR_VIEW: spoken text includes the disclaimer', unclear.lastSpoken.includes('does not decide for you'), JSON.stringify(unclear));

  // Defense-in-depth: even if the model ignores instructions and returns an
  // old-style GO/WAIT verdict, the disclaimer must still be forced in.
  const malformed = await runScenario('malformed', "GO\nTraffic has stopped, it looks clear.");
  assert('malformed model output still gets the disclaimer spoken', malformed.lastSpoken.includes('does not decide for you'), JSON.stringify(malformed));

  console.log('=== crossing mode tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
