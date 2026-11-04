// Verifies the local-VLM enrichment added after real live testing showed
// SmolVLM-500M (the local vision model) reliably produces coherent but
// shallow output for Describe mode — a flat object list with no distance,
// position, or color, since a 500M-parameter model genuinely cannot follow
// that part of the prompt the way a much larger cloud model can. Rather than
// accept that ceiling, callLocalVLM() now blends its own answer with real
// data this app already computes locally elsewhere: clock-position/proximity
// from the on-device COCO-SSD detector (localObjectSummary(), Entry 32/36)
// and dominant color (localColorGrid(), Entry 3) — genuinely local, zero
// extra cost. Tests the real function with the actual model's slow inference
// chain stubbed out (the enrichment logic itself, and the real
// localObjectSummary()/localColorGrid() calls, run for real).
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

  function stubAndCall(rawOutput, describedMode, objs){
    return `
      (async()=>{
        canvas.width=300; canvas.height=300;
        ctx.fillStyle='saddlebrown'; ctx.fillRect(0,0,300,300);
        ctx.fillStyle='forestgreen'; ctx.fillRect(0,0,150,150);
        environmentState = { objects: ${JSON.stringify(objs)}, hazards:[], pathState:'CLEAR', timestamp: Date.now() };
        localVlmReady = true;
        _tfjsModule = { load_image: async () => ({}) };
        const fakeInputs = { input_ids: { dims: { at: () => 0 } } };
        localVlmProcessor = Object.assign(async () => fakeInputs, {
          apply_chat_template: () => 'stub-text',
          batch_decode: () => [${JSON.stringify(rawOutput)}]
        });
        localVlmModel = { generate: async () => ({ slice: () => ({}) }) };
        mode = '${describedMode}';
        return await callLocalVLM('fakeimg', null, null);
      })()
    `;
  }

  // ── Describe mode with real detections: gets enriched with distance + color ──
  const r1 = await send('Runtime.evaluate', {
    expression: stubAndCall(
      'a bench, chairs, tables, and some other objects.',
      'describe',
      [{class:'bottle', clockPosition:"12 o'clock", proximity:'VERY_CLOSE'}, {class:'chair', clockPosition:"3 o'clock", proximity:'FAR'}]
    ),
    awaitPromise: true, returnByValue: true
  });
  const d1 = r1.result.value;
  assert('Describe mode with real detections is enriched with real clock-position + proximity data',
    /12 o'clock/.test(d1) && /very close/i.test(d1), d1);
  assert('Describe mode is enriched with real dominant-color data',
    /tones\.?$/i.test(d1) || /mostly/i.test(d1), d1);
  assert('the model\'s own original sentence is preserved, not replaced',
    d1.includes('a bench, chairs, tables'), d1);
  assert('a missing trailing period on the model\'s own sentence is corrected before appending',
    !/objects I can see/i.test(d1), d1); // i.e. "objects." + " I can see", not "objects I can see" run together

  // ── Describe mode with NO detections: no enrichment forced/fabricated ──
  const r2 = await send('Runtime.evaluate', {
    expression: stubAndCall('a plain white wall.', 'describe', []),
    awaitPromise: true, returnByValue: true
  });
  const d2 = r2.result.value;
  assert('with zero real detections, no fabricated object/distance claim is appended',
    !/o'clock/.test(d2), d2);

  // ── Non-describe modes are NOT enriched (scope is deliberately limited) ──
  const r3 = await send('Runtime.evaluate', {
    expression: stubAndCall(
      'a red apple on a plate.',
      'food',
      [{class:'bottle', clockPosition:"12 o'clock", proximity:'VERY_CLOSE'}]
    ),
    awaitPromise: true, returnByValue: true
  });
  const d3 = r3.result.value;
  assert('modes other than describe are left exactly as the model produced them, unenriched',
    d3 === 'a red apple on a plate.', d3);

  console.log('=== local describe enrichment tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
