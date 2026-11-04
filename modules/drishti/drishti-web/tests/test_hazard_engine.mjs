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
  await send('Page.navigate',{url:'http://localhost:5000/'});
  await sleep(1500);

  // Deterministic tests against the real functions loaded in the page — no
  // network, no camera, no LLM. Synthetic detections at known bbox/frame sizes.
  const testScript = `
(function(){
  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  // Reset tracking state between test groups
  trackedObjects = []; nextTrackId = 1; environmentState = {objects:[],hazards:[],pathState:'UNKNOWN',timestamp:0};

  // Frame is 1280x720 for all synthetic tests. buildEnvironmentState() reads its
  // frame size from the live odCanvas (sized in production by syncOdCanvasSize()
  // against the real video feed) — set it explicitly here so the synthetic boxes
  // below are measured against the same frame size they were designed for.
  odCanvas.width=1280; odCanvas.height=720;
  const FW=1280, FH=720;

  // Test 1: clock position mapping — dead center should be 12 o'clock, far left 9, far right 3.
  assert('clock: dead center is 12', bboxClockPosition([FW/2-10,300,20,20],FW) === "12 o'clock");
  assert('clock: far left is 9', bboxClockPosition([0,300,10,20],FW) === "9 o'clock");
  assert('clock: far right is 3', bboxClockPosition([FW-10,300,10,20],FW) === "3 o'clock");

  // Test 2: proximity buckets by area ratio
  const bigBox=[100,100, FW*0.7, FH*0.7]; // ~49% of frame area -> VERY_CLOSE
  const medBox=[100,100, FW*0.3, FH*0.3]; // ~9% -> MEDIUM
  const smallBox=[100,100, 30,30]; // tiny -> FAR
  assert('proximity: huge box is VERY_CLOSE', proximityFromBbox(bigBox,FW,FH).bucket==='VERY_CLOSE');
  assert('proximity: medium box is MEDIUM', proximityFromBbox(medBox,FW,FH).bucket==='MEDIUM');
  assert('proximity: tiny box is FAR', proximityFromBbox(smallBox,FW,FH).bucket==='FAR');
  assert('proximity: edge-touching box is LOW confidence', proximityFromBbox([0,0,50,50],FW,FH).confidence==='LOW');

  // Test 3: path corridor — center-lower box is in path, far-left box is not
  assert('path: center-lower is in corridor', isInPathCorridor([FW*0.45,FH*0.6,100,100],FW,FH)===true);
  assert('path: far-left-top is NOT in corridor', isInPathCorridor([0,0,50,50],FW,FH)===false);

  // Test 4: tracking — same object across 3 ticks with growing area should be "approaching"
  trackedObjects=[]; nextTrackId=1;
  let p1={class:'person', score:0.9, bbox:[500,300,100,100]};
  updateTracking([p1]);
  const id1=p1._trackId;
  let p2={class:'person', score:0.9, bbox:[500,300,140,140]};
  updateTracking([p2]);
  let p3={class:'person', score:0.9, bbox:[500,300,200,200]};
  updateTracking([p3]);
  assert('tracking: same class/overlapping box keeps same track id', p2._trackId===id1 && p3._trackId===id1);
  assert('tracking: growing box reads as approaching', movementFor(id1)==='approaching');

  // Test 5: a NEW object (no prior track) should be flagged isNew via buildEnvironmentState
  trackedObjects=[]; nextTrackId=1;
  const firstState = buildEnvironmentState([{class:'chair', score:0.8, bbox:[600,400,80,80]}]);
  assert('env: first-seen object is isNew', firstState.objects[0].isNew===true);
  const secondState = buildEnvironmentState([{class:'chair', score:0.8, bbox:[605,405,80,80]}]);
  assert('env: same object next tick is NOT new', secondState.objects[0].isNew===false);

  // Test 6: hazard scoring — a close, approaching person in-path should be CRITICAL or HIGH;
  // a far, non-hazard-class object (e.g. a book — not in HAZARD_CLASSES) should score nothing.
  trackedObjects=[]; nextTrackId=1;
  let closePerson={class:'person', score:0.9, bbox:[FW*0.45,FH*0.5, FW*0.5, FH*0.5]}; // huge, centered, in path
  updateTracking([closePerson]);
  closePerson={class:'person', score:0.9, bbox:[FW*0.44,FH*0.49, FW*0.55, FH*0.55]}; // still growing
  updateTracking([closePerson]);
  const hazard = scoreHazard(closePerson, FW, FH);
  assert('hazard: close centered person scores HIGH or CRITICAL', hazard && (hazard.riskLevel==='CRITICAL'||hazard.riskLevel==='HIGH'), JSON.stringify(hazard));

  const nonHazard = scoreHazard({class:'book', score:0.9, bbox:[10,10,20,20]}, FW, FH);
  assert('hazard: non-hazard-class object (book) scores null', nonHazard===null);

  // Test 7: distant, non-path person should score lower than the close in-path one above
  trackedObjects=[]; nextTrackId=1;
  let farPerson={class:'person', score:0.9, bbox:[10,10,40,40]}; // tiny, corner, not in path
  updateTracking([farPerson]);
  const farHazard = scoreHazard(farPerson, FW, FH);
  assert('hazard: far corner person scores lower than close centered person', farHazard && hazard && farHazard.score < hazard.score, JSON.stringify({far:farHazard,close:hazard}));

  // Test 8: priority engine dedup — same hazard announced twice within 8s should only speak once.
  let speakCallCount=0;
  const _origSpeak=window.speak;
  window.speak=function(){ speakCallCount++; };
  lastAnnouncedHazardKey=null; lastAnnouncedHazardTime=0;
  guardianOn=true;
  const dangerState={hazards:[{object:'car',riskLevel:'CRITICAL',clockPosition:"12 o'clock",movement:'approaching'}]};
  priorityAnnounce(dangerState);
  priorityAnnounce(dangerState); // immediate repeat — should be deduped
  assert('priority: repeated identical hazard is deduped (spoke once, not twice)', speakCallCount===1);
  window.speak=_origSpeak;
  guardianOn=false;

  // Test 9: priority engine respects guardianOn gate
  speakCallCount=0;
  window.speak=function(){ speakCallCount++; };
  lastAnnouncedHazardKey=null; lastAnnouncedHazardTime=0;
  guardianOn=false;
  priorityAnnounce(dangerState);
  assert('priority: does not speak when guardianOn is false', speakCallCount===0);
  window.speak=_origSpeak;

  // Test 10: pathState reflects a blocking close object vs an all-clear frame
  trackedObjects=[]; nextTrackId=1;
  const blockedState=buildEnvironmentState([{class:'chair', score:0.9, bbox:[FW*0.4,FH*0.5,FW*0.4,FH*0.4]}]);
  assert('path: close centered obstacle -> OBSTACLE_AHEAD', blockedState.pathState==='OBSTACLE_AHEAD');
  trackedObjects=[]; nextTrackId=1;
  const clearState=buildEnvironmentState([]);
  assert('path: no detections -> UNCERTAIN (never claims CLEAR with no data)', clearState.pathState==='UNCERTAIN');

  return JSON.stringify(results);
})()
`;
  const r = await send('Runtime.evaluate', { expression: testScript, returnByValue: true });
  if (r.exceptionDetails) {
    console.log('EXCEPTION DURING TEST RUN:', r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    process.exit(1);
  }
  const results = JSON.parse(r.result.value);
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.detail ? ' :: ' + t.detail : ''));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
