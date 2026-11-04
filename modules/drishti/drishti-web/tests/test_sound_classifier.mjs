// Verifies the local sound-hazard classifier (YAMNet, TensorFlow.js) added
// after the user asked for real-world sound awareness (sirens, horns,
// alarms), independent of the camera and of callGroq()'s whole cloud
// fallback chain. Two tiers, following test_depth_engine.mjs's established
// pattern: fast deterministic logic checks (no model/mic needed) plus one
// slower real end-to-end pass with the actual model and a real (synthetic)
// audio signal, skippable via --skip-real for quick iteration.
//
// Full end-to-end verification (real mic capture -> resample -> inference ->
// announcement -> cooldown) was also done manually against a real synthetic
// siren WAV fed through Chrome's --use-file-for-fake-audio-capture flag (see
// tests/README.md) — that's the authoritative proof this actually works
// live; this file covers what CAN be reliably automated (the deterministic
// logic, plus a real model load+inference check).
const CDP = 'http://localhost:9333';
const skipReal = process.argv.includes('--skip-real');
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

  // ── Fast logic: allowlist + threshold + cooldown/dedup (no model/mic needed) ──
  const r1 = await send('Runtime.evaluate', {
    expression: `
      (function(){
        const sysMsgs=[]; const spoken=[];
        const origAddMsg=addMsg, origSpeak=speak;
        window.addMsg=(role,text)=>{ if(role==='sys') sysMsgs.push(text); };
        window.speak=(t)=>{ spoken.push(t); };
        lastAnnouncedSoundKey=null; lastAnnouncedSoundTime=0;
        announceSound(390, 0.5); // 'a siren' — well above threshold
        const firstCallCount = sysMsgs.length;
        announceSound(390, 0.5); // same sound, immediately again — should be suppressed by cooldown
        const secondCallCount = sysMsgs.length;
        window.addMsg=origAddMsg; window.speak=origSpeak;
        return JSON.stringify({ firstCallCount, secondCallCount, sysMsgs, spoken });
      })()
    `,
    returnByValue: true
  });
  const d1 = JSON.parse(r1.result.value);
  assert('a relevant sound above threshold triggers exactly one announcement',
    d1.firstCallCount === 1 && d1.spoken.length === 1, JSON.stringify(d1));
  assert('the same sound repeated immediately is suppressed by the cooldown, not re-announced',
    d1.secondCallCount === 1, JSON.stringify(d1));
  assert('the announcement names the real sound and never claims a direction (a mic cannot localize)',
    /siren/i.test(d1.sysMsgs[0]) && !/o'clock/i.test(d1.sysMsgs[0]), JSON.stringify(d1));

  // ── A different sound class right after should NOT be suppressed (dedup is per-key) ──
  const r2 = await send('Runtime.evaluate', {
    expression: `
      (function(){
        const sysMsgs=[];
        const origAddMsg=addMsg;
        window.addMsg=(role,text)=>{ if(role==='sys') sysMsgs.push(text); };
        lastAnnouncedSoundKey=null; lastAnnouncedSoundTime=0;
        announceSound(390, 0.5); // siren
        announceSound(304, 0.5); // car alarm — different class, should still announce
        window.addMsg=origAddMsg;
        return JSON.stringify({ count: sysMsgs.length, sysMsgs });
      })()
    `,
    returnByValue: true
  });
  const d2 = JSON.parse(r2.result.value);
  assert('a genuinely different sound class is not suppressed by another class\'s cooldown',
    d2.count === 2, JSON.stringify(d2));

  // ── The allowlist itself: sanity-check a few real, confirmed indices ──
  const r3 = await send('Runtime.evaluate', {
    expression: `JSON.stringify({
      siren: SOUND_ALLOWLIST[390], alarm: SOUND_ALLOWLIST[382],
      ambulance: SOUND_ALLOWLIST[318], carAlarm: SOUND_ALLOWLIST[304],
      excludesAlarmClock: SOUND_ALLOWLIST[389], excludesAnimalHonk: SOUND_ALLOWLIST[102]
    })`,
    returnByValue: true
  });
  const d3 = JSON.parse(r3.result.value);
  assert('the allowlist maps the real, confirmed YAMNet indices correctly',
    d3.siren==='a siren' && d3.alarm==='an alarm' && d3.ambulance==='an ambulance siren' && d3.carAlarm==='a car alarm',
    JSON.stringify(d3));
  assert('the allowlist deliberately excludes alarm clock (domestic) and animal honk (not vehicles)',
    d3.excludesAlarmClock===undefined && d3.excludesAnimalHonk===undefined, JSON.stringify(d3));

  if (!skipReal) {
    // ── Slow, real: actually load YAMNet and run one real inference ──
    console.log('Loading real YAMNet model (downloads on first run, can take 30-60s)...');
    const r4 = await send('Runtime.evaluate', {
      expression: `
        (async()=>{
          await loadSoundClassifier();
          if(!soundReady) return JSON.stringify({ ok:false, reason:'model failed to load' });
          const sr = 16000;
          const samples = new Float32Array(sr);
          for(let i=0;i<sr;i++) samples[i] = Math.sin(2*Math.PI*440*i/sr) * 0.5;
          const input = tf.tensor1d(samples);
          const output = soundModel.execute(input);
          const outArr = Array.isArray(output) ? output : [output];
          const shapesOk = outArr.length === 3 && outArr[0].shape[1] === 521;
          tf.dispose([input, ...outArr]);
          return JSON.stringify({ ok:true, shapesOk, numOutputs: outArr.length });
        })()
      `,
      awaitPromise: true, returnByValue: true, timeout: 120000
    });
    const d4 = JSON.parse(r4.result.value);
    assert('the real model loads and produces the confirmed real output shape ([N,521] scores + embeddings + spectrogram)',
      d4.ok && d4.shapesOk, JSON.stringify(d4));
  } else {
    console.log('Skipping real model load (--skip-real).');
  }

  console.log('\n=== sound classifier tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
