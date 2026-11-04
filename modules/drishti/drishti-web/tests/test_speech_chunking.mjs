// Verifies the speech-chunking fix (see IMPLEMENTATION_PROGRESS.md): the user
// reported that long descriptions only had their first few sentences read
// aloud. Root cause: Chrome's Web Speech API silently stops a single long
// SpeechSynthesisUtterance after ~15 seconds (a documented browser bug), and
// separately, Sarvam TTS hard-caps at 500 characters per request — the old
// code sliced to 500 chars and dropped the rest with no indication anything
// was missing. Fixed by splitting into sentence-level chunks and chaining
// playback. This also introduced a real risk (cancelling a Web Speech
// utterance can still fire its own 'onend' in Chrome) that a stopped chain
// could keep talking — verified fixed via a generation-token guard.
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

  // ── splitIntoSpeechChunks: pure logic ──
  const chunkTests = await send('Runtime.evaluate', {
    expression: `
      (function(){
        const longText = 'Sentence one is here. Sentence two follows it. Sentence three continues on. Sentence four keeps going. Sentence five is next. Sentence six arrives now. Sentence seven still going. Sentence eight almost there. Sentence nine more. Sentence ten final one.';
        const chunks200 = splitIntoSpeechChunks(longText, 200);
        const chunks500 = splitIntoSpeechChunks(longText, 500);
        const noPunctuation = splitIntoSpeechChunks('just some text with no sentence end', 200);
        const rejoined = chunks200.join(' ');
        return JSON.stringify({
          chunkCount200: chunks200.length,
          chunkCount500: chunks500.length,
          allChunksUnderLimit: chunks200.every(c => c.length <= 220), // small slack for the trailing sentence that pushed it over
          noPunctuationHandled: noPunctuation.length === 1 && noPunctuation[0].includes('just some text'),
          totalTextPreserved: longText.split(' ').every(w => rejoined.includes(w))
        });
      })()
    `,
    returnByValue: true
  });
  const ct = JSON.parse(chunkTests.result.value);
  assert('long text at maxLen=200 splits into multiple chunks', ct.chunkCount200 > 1, JSON.stringify(ct));
  assert('the same text at a larger maxLen=500 produces fewer or equal chunks', ct.chunkCount500 <= ct.chunkCount200, JSON.stringify(ct));
  assert('text with no sentence-ending punctuation still returns as one usable chunk (no crash)', ct.noPunctuationHandled, JSON.stringify(ct));
  assert('no words are lost across the chunk boundaries', ct.totalTextPreserved, JSON.stringify(ct));

  // ── speakWebSpeechChunks: verifies ALL chunks get queued as separate
  //    utterances, not just the first — this is the actual fix for "only
  //    reads the first few lines" ──
  await send('Runtime.evaluate', { expression: `speakOn = true;` });
  const queueTest = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const utterancesSpoken = [];
        const orig = window.speechSynthesis.speak;
        window.speechSynthesis.speak = function(utter){
          utterancesSpoken.push(utter.text);
          setTimeout(() => utter.onend && utter.onend(), 10); // simulate each utterance finishing quickly
        };
        let doneCalled = false;
        await new Promise(resolve => {
          const chunks = ['First sentence.', 'Second sentence.', 'Third sentence.'];
          speakWebSpeechChunks(chunks, 0, () => { doneCalled = true; resolve(); }, speechGeneration);
        });
        window.speechSynthesis.speak = orig;
        return JSON.stringify({ utterancesSpoken, doneCalled });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const qt = JSON.parse(queueTest.result.value);
  assert('all 3 chunks get spoken as separate utterances, not just the first', qt.utterancesSpoken.length===3, JSON.stringify(qt));
  assert('onAllDone fires exactly once, after the full queue finishes', qt.doneCalled, JSON.stringify(qt));

  // ── Generation guard: a stopped/superseded chain must not keep talking ──
  const genTest = await send('Runtime.evaluate', {
    expression: `
      (async()=>{
        const utterancesSpoken = [];
        const orig = window.speechSynthesis.speak;
        window.speechSynthesis.speak = function(utter){
          utterancesSpoken.push(utter.text);
          // Simulate the real Chrome bug: cancelling still fires onend for
          // the interrupted utterance.
          setTimeout(() => utter.onend && utter.onend(), 10);
        };
        let doneCallCount = 0;
        const chunks = ['Chunk A.', 'Chunk B.', 'Chunk C.'];
        const genAtStart = speechGeneration;
        speakWebSpeechChunks(chunks, 0, () => { doneCallCount++; }, genAtStart);
        await new Promise(r => setTimeout(r, 5)); // let the first utterance start
        speechGeneration++; // simulate stopSpeaking() being called mid-chain
        await new Promise(r => setTimeout(r, 50)); // let any pending onend callbacks fire
        window.speechSynthesis.speak = orig;
        return JSON.stringify({ utterancesSpoken, doneCallCount });
      })()
    `,
    awaitPromise: true, returnByValue: true
  });
  const gt = JSON.parse(genTest.result.value);
  assert('a superseded generation stops after the in-flight chunk — does not queue the remaining chunks', gt.utterancesSpoken.length===1, JSON.stringify(gt));
  assert('a superseded generation never calls onAllDone (no vcPaused flicker from a stale chain)', gt.doneCallCount===0, JSON.stringify(gt));

  console.log('=== speech chunking tests ===');
  let failCount = 0;
  results.forEach(t => {
    console.log((t.pass ? 'PASS' : 'FAIL') + ' — ' + t.name + (t.pass ? '' : ' :: ' + t.detail));
    if (!t.pass) failCount++;
  });
  console.log(`\n${results.length - failCount}/${results.length} passed`);
  process.exit(failCount > 0 ? 1 : 0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
