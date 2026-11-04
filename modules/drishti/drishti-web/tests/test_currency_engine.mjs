// Verifies local currency recognition (OCR + color heuristic — not a trained
// classifier, see ARCHITECTURE_AUDIT.md §9). Tests the deterministic extraction
// logic directly, then confirms the full pipeline with real OCR against a
// canvas rendered to look like a note (denomination number + background color).
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

  // Pure logic: extractDenomination — no OCR/network involved.
  const logicTests = await send('Runtime.evaluate', {
    expression: `
      JSON.stringify({
        clearFiveHundred: extractDenomination('RESERVE BANK OF INDIA 500 rupees 500 Mahatma Gandhi 500'),
        noMatch: extractDenomination('RANDOM TEXT WITH NO VALID DENOMINATION 37 42'),
        avoidsSerialNumberFalsePositive: extractDenomination('7AK 500123 RESERVE BANK OF INDIA'),
        doesNotConfuse100With1000: extractDenomination('this is 1000 not the note'),
        picksMostFrequent: extractDenomination('100 100 100 200')
      })
    `,
    returnByValue: true
  });
  const lt = JSON.parse(logicTests.result.value);
  assert('extracts ₹500 when it appears multiple times', lt.clearFiveHundred && lt.clearFiveHundred.denomination===500 && lt.clearFiveHundred.occurrences>=2, JSON.stringify(lt.clearFiveHundred));
  assert('returns null when no valid denomination text is present', lt.noMatch===null, JSON.stringify(lt.noMatch));
  assert('a lone number embedded in a serial-number-like string does not count as 2+ occurrences', !lt.avoidsSerialNumberFalsePositive || lt.avoidsSerialNumberFalsePositive.occurrences<2, JSON.stringify(lt.avoidsSerialNumberFalsePositive));
  assert('word-boundary regex does not match ₹100 inside "1000"', lt.doesNotConfuse100With1000===null, JSON.stringify(lt.doesNotConfuse100With1000));
  assert('picks the most frequently occurring denomination when multiple are present', lt.picksMostFrequent && lt.picksMostFrequent.denomination===100, JSON.stringify(lt.picksMostFrequent));

  // Color-hint cross-check logic (pure, no network)
  const colorTests = await send('Runtime.evaluate', {
    expression: `
      JSON.stringify({
        brownMatchesTen: CURRENCY_COLOR_HINTS[10].includes('brown'),
        grayMatchesFiveHundred: CURRENCY_COLOR_HINTS[500].includes('gray'),
        magentaMatchesTwoThousand: CURRENCY_COLOR_HINTS[2000].includes('magenta')
      })
    `,
    returnByValue: true
  });
  const ct = JSON.parse(colorTests.result.value);
  assert('₹10 color hints include brown (real note color)', ct.brownMatchesTen);
  assert('₹500 color hints include gray (real note color)', ct.grayMatchesFiveHundred);
  assert('₹2000 color hints include magenta (real note color)', ct.magentaMatchesTwoThousand);

  // Full pipeline with real OCR: render a synthetic "note" (gray background,
  // "500" printed multiple times) and confirm localCurrencyRead() correctly
  // identifies it as HIGH confidence; then a color MISMATCH case (right number,
  // wrong color) should never be trusted (falls back to cloud in the real app).
  console.log('Loading OCR worker (real Tesseract, first run downloads WASM)...');
  await send('Runtime.evaluate', { expression: `loadOcrWorker()`, awaitPromise: true });
  await sleep(500);

  const matchCase = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        canvas.width=800; canvas.height=400;
        ctx.fillStyle='rgb(130,130,130)'; ctx.fillRect(0,0,800,400); // gray, like a real ₹500 note
        ctx.fillStyle='black'; ctx.font='bold 60px Arial';
        ctx.fillText('500', 40, 100);
        ctx.fillText('RESERVE BANK', 40, 200);
        ctx.fillText('500', 40, 300);
        const r = await localCurrencyRead();
        return JSON.stringify(r);
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const match = JSON.parse(matchCase.result.value);
  assert('matching number+color: correctly identifies ₹500 at HIGH confidence', match && match.denomination===500 && match.confidence==='HIGH', JSON.stringify(match));

  const mismatchCase = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        canvas.width=800; canvas.height=400;
        ctx.fillStyle='rgb(230,30,30)'; ctx.fillRect(0,0,800,400); // red background — wrong color for any real note, esp. ₹500 (gray)
        ctx.fillStyle='black'; ctx.font='bold 60px Arial';
        ctx.fillText('500', 40, 100);
        ctx.fillText('RESERVE BANK', 40, 200);
        ctx.fillText('500', 40, 300);
        const r = await localCurrencyRead();
        return JSON.stringify(r);
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const mismatch = JSON.parse(mismatchCase.result.value);
  assert('number matches but color does not: confidence is MEDIUM, not HIGH (never silently trusted)', mismatch && mismatch.confidence==='MEDIUM', JSON.stringify(mismatch));

  console.log('\n=== currency engine tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
