// Verifies a real bug the user found by actually using voice commands: "it
// does not even listen properly." Root cause (found by code review, not
// guessed): if the camera is stopped (or never started) while voice commands
// remain toggled on, a spoken mode command like "describe" matched its regex
// perfectly, then called setModeVoice()/snap() — both of which silently do
// nothing without an active camera stream. No sound, no message, nothing. A
// sighted user might notice the mode pill still changed; a blind user gets
// total silence and has no way to know the command was even heard. Fixed by
// speaking "Start the camera first" instead of failing silently, matching the
// pattern already used elsewhere in this same function (streetNavigate,
// crossing/translate/gesture) but missing from the main CMD_MAP/REGIONAL_CMDS
// mode-matching loops.
//
// Separately: `vcPaused` (true while Drishti is speaking, so the command mic
// doesn't hear its own voice) is a real, independent mechanism covered by its
// own tests elsewhere. This file forces it false before each check so a slow
// or overlapping startup announcement in the test's headless TTS can't create
// a flaky race — that's not what's being tested here.
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

  async function checkNoCamFeedback(cmdText, expectedPattern){
    const r = await send('Runtime.evaluate', {
      expression: `
        (async()=>{
          try{stopCam();}catch(e){}
          vcPaused=false; micOn=false; lastCmdTime=0;
          const spoken=[];
          const orig=speak;
          window.speak=(t)=>{spoken.push(t);};
          const modeBefore=mode;
          let threw=null;
          try{ await handleCmd(${JSON.stringify(cmdText)}); }catch(e){ threw=e.message; }
          await new Promise(r=>setTimeout(r,50));
          window.speak=orig;
          return JSON.stringify({spoken, modeChanged: mode!==modeBefore, streamNow: !!stream, threw});
        })()
      `,
      awaitPromise: true, returnByValue: true
    });
    return JSON.parse(r.result.value);
  }

  const d1 = await checkNoCamFeedback('describe');
  assert('English CMD_MAP mode command with no camera running gets real spoken feedback, not silence',
    d1.spoken.some(s=>/start the camera/i.test(s)) && !d1.modeChanged, JSON.stringify(d1));

  const d2 = await checkNoCamFeedback('बताओ');
  assert('Hindi REGIONAL_CMDS mode command with no camera running also gets spoken feedback, not silence',
    d2.spoken.some(s=>/camera/i.test(s)) && !d2.modeChanged, JSON.stringify(d2));

  const d3 = await checkNoCamFeedback('take a photo');
  assert('a generic snap-only command (mode:null) with no camera running also gets spoken feedback',
    d3.spoken.some(s=>/start the camera/i.test(s)), JSON.stringify(d3));

  // Sanity check the fix doesn't regress the working case: with the camera
  // genuinely running, "describe" should still work exactly as before —
  // change mode and NOT speak a "start the camera" message.
  const r4 = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        await startCam();
        await new Promise(r=>setTimeout(r,300));
        vcPaused=false; micOn=false; lastCmdTime=0;
        const spoken=[];
        const orig=speak;
        window.speak=(t)=>{spoken.push(t);};
        const modeBefore=mode;
        let threw=null;
        try{ await handleCmd('read this'); }catch(e){ threw=e.message; }
        await new Promise(r=>setTimeout(r,50));
        window.speak=orig;
        return JSON.stringify({spoken, modeChanged: mode!==modeBefore, modeNow: mode, streamNow: !!stream, threw});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const d4 = JSON.parse(r4.result.value);
  assert('with the camera actually running, a mode command still switches mode and never gets the false-positive "start camera" message',
    d4.streamNow && d4.modeChanged && d4.modeNow==='text' && !d4.spoken.some(s=>/start the camera/i.test(s)), JSON.stringify(d4));

  console.log('=== voice command feedback tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
