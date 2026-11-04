// Verifies local OCR (Tesseract.js) actually runs and actually recognizes text —
// not a mocked response. Renders real text onto the app's own capture canvas
// with the browser's Canvas API, then checks the real OCR engine reads it back.
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

  console.log('Loading Tesseract.js (downloads WASM + language data on first run — can take up to a minute)...');
  const loadResult = await send('Runtime.evaluate', {
    expression: `(async()=>{ await loadOcrWorker(); return JSON.stringify({ocrReady, hasWorker: !!ocrWorker}); })()`,
    awaitPromise: true, returnByValue: true
  });
  console.log('Load result:', loadResult.result?.value, loadResult.exceptionDetails?.text || '');

  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  const loadState = JSON.parse(loadResult.result.value);
  assert('Tesseract worker loads successfully', loadState.ocrReady && loadState.hasWorker, JSON.stringify(loadState));

  if (loadState.ocrReady) {
    // Render real, clear, high-contrast text onto the app's own capture canvas,
    // then run the real local OCR function on it — genuine recognition, not a stub.
    const ocrResult = await send('Runtime.evaluate', {
      expression: `
        (async()=>{
          canvas.width=600; canvas.height=200;
          ctx.fillStyle='white'; ctx.fillRect(0,0,600,200);
          ctx.fillStyle='black'; ctx.font='bold 48px Arial';
          ctx.fillText('HELLO WORLD', 40, 100);
          const r = await localOCR(canvas);
          return JSON.stringify(r);
        })()
      `,
      awaitPromise: true, returnByValue: true
    });
    const ocr = JSON.parse(ocrResult.result.value);
    assert('OCR returns a result object', !!ocr, JSON.stringify(ocr));
    assert('OCR recognizes the rendered text (case/spacing tolerant)',
      ocr && /hello/i.test(ocr.text) && /world/i.test(ocr.text), JSON.stringify(ocr));
    assert('OCR reports a confidence score above the app\'s threshold for clean text',
      ocr && ocr.confidence >= 65, JSON.stringify(ocr));

    // Blank canvas — no text at all. Local OCR should not fabricate text; low/no
    // confidence here is what makes snap() correctly fall back to the cloud model.
    const blankResult = await send('Runtime.evaluate', {
      expression: `
        (async()=>{
          canvas.width=600; canvas.height=200;
          ctx.fillStyle='white'; ctx.fillRect(0,0,600,200);
          const r = await localOCR(canvas);
          return JSON.stringify(r);
        })()
      `,
      awaitPromise: true, returnByValue: true
    });
    const blank = JSON.parse(blankResult.result.value);
    assert('blank image: OCR does not fabricate substantial text', blank && blank.text.trim().length < 3, JSON.stringify(blank));
  } else {
    assert('OCR content tests skipped (worker failed to load — likely no network to the CDN in this environment)', true, 'skipped');
  }

  console.log('\n=== OCR tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
