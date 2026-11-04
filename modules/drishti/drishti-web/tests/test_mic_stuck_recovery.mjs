// Verifies the fix for a real, high-impact bug found from a live "I say the
// same thing 10 times and nothing happens at all" report: micOn (set by the
// separate "Ask Drishti" chat mic) had no timeout safety net — if its
// recognition.onend event ever failed to fire (a real, known class of Web
// Speech API flakiness — the same family of bug vcPaused's own Entry-32 fix
// already accounts for), micOn stayed stuck true forever, and handleCmd()
// silently discarded every single voice command from then on, with zero
// error and the recognizer itself still running/restarting normally — so
// nothing looked broken from the outside.
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
  await send('Runtime.evaluate', { expression: `try{stopCam();}catch(e){}` });

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  // ── A stuck micOn (>15s old) is auto-recovered and the command proceeds ──
  const r1 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        try{stopCam();}catch(e){}
        const sysMsgs=[];
        const spoken=[];
        const origAddMsg=addMsg, origSpeak=speak;
        window.addMsg=(role,text)=>{ if(role==='sys') sysMsgs.push(text); };
        window.speak=(t)=>{ spoken.push(t); };
        micOn=true; micOnAt=Date.now()-16000; // simulate a stuck mic from 16s ago
        vcPaused=false;
        await handleCmd('describe');
        const micOnAfter=micOn;
        window.addMsg=origAddMsg; window.speak=origSpeak;
        return JSON.stringify({ micOnAfter, sysMsgs, spoken });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d1 = JSON.parse(r1.result.value);
  assert('a stuck micOn older than 15s is auto-cleared',
    d1.micOnAfter === false, JSON.stringify(d1));
  assert('a real system message explains the recovery',
    d1.sysMsgs.some(s=>/mic got stuck/i.test(s)), JSON.stringify(d1));
  assert('the command that triggered recovery is NOT silently dropped — it proceeds (camera not running, so "start the camera" feedback, not silence)',
    d1.spoken.some(s=>/start the camera/i.test(s)), JSON.stringify(d1));

  // ── A recently-set micOn (genuinely mid-conversation) is NOT force-cleared ──
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const sysMsgs=[];
        const origAddMsg=addMsg;
        window.addMsg=(role,text)=>{ if(role==='sys') sysMsgs.push(text); };
        await new Promise(r=>setTimeout(r,3100)); // clear the 3s inter-command cooldown from test 1
        micOn=true; micOnAt=Date.now()-2000; // only 2s old — genuinely still in progress
        vcPaused=false;
        await handleCmd('describe');
        const micOnAfter=micOn;
        window.addMsg=origAddMsg;
        return JSON.stringify({ micOnAfter, sysMsgs });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('a recent (genuinely in-progress) micOn is left alone, not force-cleared',
    d2.micOnAfter === true, JSON.stringify(d2));
  assert('no recovery message fires for a genuinely-in-progress mic session',
    !d2.sysMsgs.some(s=>/mic got stuck/i.test(s)), JSON.stringify(d2));

  console.log('=== mic-stuck recovery tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
