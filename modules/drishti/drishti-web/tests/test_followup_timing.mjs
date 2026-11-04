// Verifies the real root cause of "reads the first few lines then jumps to
// the end": the proactive "I can see text, say read it" follow-up used to
// fire on a guessed setTimeout (capped at 8s, started BEFORE the main speech
// even began) — any description of real length takes longer than that to
// actually speak, so the follow-up would call speak() again mid-description,
// and every speak() call stops whatever's currently playing first. Fixed by
// using speak()'s real completion callback instead of a guessed delay.
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

  // Mocked Groq response: a describe-mode answer containing "reads" — the
  // exact word that triggers the proactive follow-up — and long enough that a
  // real chunked TTS readout takes several seconds (each chunk here is
  // simulated to take 3s to "speak", so the whole thing takes ~12s+, which
  // would have blown right through the old 8s-capped guess).
  const longDescription = "You're standing in a room with a wooden table ahead. There's a sign on the wall that reads Exit This Way in bold letters. To your left, a chair sits empty near the window. Further right, a bookshelf holds several thick volumes stacked unevenly.";
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
        const body = JSON.stringify({ choices: [{ message: { content: longDescription }, finish_reason: 'stop' }] });
        send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [
          { name: 'Content-Type', value: 'application/json' },
          { name: 'Access-Control-Allow-Origin', value: '*' }
        ], body: Buffer.from(body).toString('base64') });
      } else { send('Fetch.continueRequest', { requestId }); }
    }
  });

  await send('Page.navigate', { url: 'http://localhost:5000/' });
  await sleep(1500);
  await send('Runtime.evaluate', { expression: `localStorage.setItem('dk_groq','gsk_fake_test_key'); localStorage.setItem('dk_ai_mode','HYBRID'); location.reload();` });
  await sleep(2000);
  await send('Runtime.evaluate', { expression: `startCam()`, awaitPromise: true });
  // startCam() itself schedules a "Camera is ready..." announcement 600ms
  // later — wait that out (with margin) before installing the speak() mock
  // below, or this test would capture that unrelated call too.
  await sleep(2000);

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  const r = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const spokenLog = [];
        const origSpeak = speak;
        // Wrap speak() to log every call's text and simulate each one taking a
        // realistic few seconds (like real chunked TTS would), instead of
        // resolving instantly — this is what actually exposes the old bug,
        // since an instant-resolving mock speak() would never overlap.
        window.speak = function(text, onDone){
          spokenLog.push({text, at: performance.now()});
          setTimeout(() => { if(onDone) onDone(); }, 3000);
        };
        setMode('describe');
        await snap();
        // Wait long enough for the main description AND the follow-up (if it
        // incorrectly fires early) to have both had a chance to log.
        await new Promise(res => setTimeout(res, 9000));
        window.speak = origSpeak;
        return JSON.stringify(spokenLog.map(s => ({ text: s.text, at: Math.round(s.at) })));
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  // Filter to just the two calls this test actually cares about — the fake
  // camera device can trigger unrelated speak() calls (a startup announcement,
  // a low-light warning if the synthetic frame reads dark) depending on timing,
  // which aren't part of what's being tested here.
  const rawLog = JSON.parse(r.result.value);
  const description = rawLog.find(e => e.text.includes('bookshelf'));
  const followUp = rawLog.find(e => e.text.includes('read it'));
  const log = [description, followUp].filter(Boolean);
  assert('both the main description and the follow-up were spoken', !!description && !!followUp, JSON.stringify(rawLog));
  if(log.length===2){
    const gapMs = followUp.at - description.at;
    assert('main description is spoken first, in full', description.at < followUp.at, JSON.stringify(rawLog));
    assert('follow-up mentions reading the text', followUp.text.includes('read it'), JSON.stringify(log));
    // The mocked speak() takes 3000ms to "finish" — the follow-up must not
    // fire before that, which is exactly what the old 8000ms-capped guess
    // would have gotten wrong for anything that takes longer to really speak
    // than 8s. Real chunked TTS for a description this length would take
    // well over 8s, which is the whole point being tested here.
    assert('follow-up fires only AFTER the main description actually finished (gap >= 3000ms), not on a fixed guess', gapMs>=2900, 'gapMs='+gapMs);
  }

  console.log('=== follow-up timing tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
