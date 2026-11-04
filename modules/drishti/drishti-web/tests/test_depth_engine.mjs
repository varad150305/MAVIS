// Verifies real monocular depth estimation (Xenova/depth-anything-small-hf via
// Transformers.js — see ARCHITECTURE_AUDIT.md and the module comment above
// refineProximityWithDepth in index.html for why this is on-demand only, not
// continuous). Two tiers: fast deterministic tests against synthetic depth
// maps (no model, no network — pure sampling/bucketing logic), then one real
// end-to-end run of the actual model, which is slow (~15-90s incl. first-time
// model download) but proves the real integration genuinely works, not just
// the logic around it.
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

  // ── Fast deterministic tests: sampleDepthInBbox against a synthetic map ──
  const logicTests = await send('Runtime.evaluate', {
    expression: `
      (function(){
        // 100x100 depth map: left half depth=200 (near), right half depth=50 (far)
        const w=100, h=100;
        const data = new Float32Array(w*h);
        for(let y=0;y<h;y++) for(let x=0;x<w;x++) data[y*w+x] = x<50 ? 200 : 50;
        const map = {data, width:w, height:h};

        // A bbox entirely in the "near" (left) half of a 200x200 video frame
        // (which maps 1:1 in scale to the 100x100 depth map: frameW/depthW=2)
        const nearBbox = [10,10,50,50]; // in 200x200 frame coords -> depth coords [5,5,25,25], all in left half
        const nearSample = sampleDepthInBbox(map, nearBbox, 200, 200);

        // A bbox entirely in the "far" (right) half
        const farBbox = [150,10,40,40]; // -> depth coords [75,5,20,20], all in right half
        const farSample = sampleDepthInBbox(map, farBbox, 200, 200);

        return JSON.stringify({nearSample, farSample});
      })()
    `,
    returnByValue: true
  });
  const lt = JSON.parse(logicTests.result.value);
  assert('sampleDepthInBbox: near region samples the near depth value (200)', lt.nearSample===200, JSON.stringify(lt));
  assert('sampleDepthInBbox: far region samples the far depth value (50)', lt.farSample===50, JSON.stringify(lt));

  // ── refineProximityWithDepth's bucketing, with estimateDepthMap mocked to
  //    avoid needing the real model for this fast-path logic test ──
  const bucketTests = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const origEstimate = estimateDepthMap;
        // Whole frame ranges 0-100; test bbox always samples the same region,
        // varied via a controllable synthetic map per call.
        function makeMap(objDepthValue){
          const w=10,h=10;
          const data = new Float32Array(w*h).fill(0); // rest of frame = far (0)
          // bbox below maps to roughly the top-left 2x2 cells of a 10x10 map
          // scaled from a 100x100 frame (bbox [0,0,20,20] -> depth coords [0,0,2,2])
          for(let y=0;y<2;y++) for(let x=0;x<2;x++) data[y*w+x]=objDepthValue;
          return {data, width:w, height:h};
        }
        const bbox=[0,0,20,20], frameW=100, frameH=100;

        estimateDepthMap = async () => makeMap(100); // object itself is the closest thing in frame -> VERY_CLOSE
        const veryClose = await refineProximityWithDepth(bbox, frameW, frameH);

        estimateDepthMap = async () => { const m=makeMap(0); for(let i=0;i<m.data.length;i++) if(m.data[i]===0) m.data[i]=0; return m; }; // object is part of the uniformly-far background -> should be FAR-ish, not confidently close
        const uniform = await refineProximityWithDepth(bbox, frameW, frameH);

        estimateDepthMap = async () => null; // model unavailable
        const unavailable = await refineProximityWithDepth(bbox, frameW, frameH);

        estimateDepthMap = origEstimate;
        return JSON.stringify({veryClose, uniform, unavailable});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const bt = JSON.parse(bucketTests.result.value);
  assert('refineProximityWithDepth: object depth far above frame average -> VERY_CLOSE, HIGH confidence', bt.veryClose && bt.veryClose.bucket==='VERY_CLOSE' && bt.veryClose.confidence==='HIGH', JSON.stringify(bt.veryClose));
  assert('refineProximityWithDepth: object at the same depth as a uniform background -> not VERY_CLOSE', bt.uniform && bt.uniform.bucket!=='VERY_CLOSE', JSON.stringify(bt.uniform));
  assert('refineProximityWithDepth: returns null (not a guess) when the depth model is unavailable', bt.unavailable===null, JSON.stringify(bt.unavailable));

  // ── checkDepthForClosestHazard: falls back to the bbox-area proxy when the
  //    depth model can't be used, rather than saying nothing ──
  const fallbackTest = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        await startCam();
        environmentState = { objects: [{class:'chair', bbox:[10,10,100,100], proximity:'CLOSE'}], hazards:[], pathState:'UNKNOWN', timestamp:Date.now() };
        depthReady = false; depthModelLoading = true; // force the "unavailable" path without waiting on a real load
        window.__spoken = [];
        const _orig = speak;
        speak = (t) => { window.__spoken.push(t); };
        await checkDepthForClosestHazard();
        speak = _orig;
        depthModelLoading = false;
        return JSON.stringify(window.__spoken);
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const spoken = JSON.parse(fallbackTest.result.value);
  assert('checkDepthForClosestHazard: falls back to the bbox proxy bucket (CLOSE -> "fairly close") when depth is unavailable', spoken.some(s=>s.includes('fairly close')), JSON.stringify(spoken));
  assert('checkDepthForClosestHazard: mentions the actual object class', spoken.some(s=>s.includes('chair')), JSON.stringify(spoken));

  console.log('=== fast depth-logic tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);

  // ── One real, slow, end-to-end run of the actual model (skipped via
  //    --skip-real flag for fast CI runs) ──
  if (process.argv.includes('--skip-real')) {
    console.log('\n(skipped real model test — pass without --skip-real to run it)');
    process.exit(failCount > 0 ? 1 : 0);
  }
  console.log('\nRunning real depth model end-to-end (downloads model on first run, can take 30-90s)...');
  const realTest = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        await loadDepthModel();
        if(!depthReady) return JSON.stringify({loaded:false});
        canvas.width=200; canvas.height=200;
        ctx.fillStyle='#334455'; ctx.fillRect(0,0,200,200);
        ctx.fillStyle='#cc8844'; ctx.fillRect(50,50,100,100);
        // Feed the synthetic frame through by drawing it onto depthCanvas via video substitute:
        // estimateDepthMap reads from the live <video>, so for this deterministic
        // test we call the pipeline directly on our canvas instead.
        const dataUrl = canvas.toDataURL('image/png');
        const output = await depthModel(dataUrl);
        const map = {data: output.depth.data, width: output.depth.width, height: output.depth.height};
        const near = sampleDepthInBbox(map, [50,50,100,100], 200, 200);
        const far = sampleDepthInBbox(map, [0,0,30,30], 200, 200);
        return JSON.stringify({loaded:true, near, far, nearIsCloserThanFar: near > far});
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const real = JSON.parse(realTest.result.value);
  const realPass = real.loaded && real.nearIsCloserThanFar;
  console.log((realPass?'PASS':'FAIL')+' — real model: correctly ranks a large centered block as closer than a background corner :: '+JSON.stringify(real));

  process.exit((failCount > 0 || !realPass) ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
