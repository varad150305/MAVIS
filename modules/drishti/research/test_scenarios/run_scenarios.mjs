// Runs the 12 named scenarios in scenarios.md through the REAL, unmodified
// hazard-engine functions (scoreHazard, isInPathCorridor, proximityFromBbox,
// bboxClockPosition) inside a real running instance of drishti-web via the
// Chrome DevTools Protocol — the same technique drishti-web/tests already
// uses, and for the same reason: this exercises the actual production code,
// not a reimplementation of its logic. Nothing here is a pre-computed or
// guessed result; every number in the output was produced by evaluating the
// real function against the real (synthetic, but honestly labeled as such)
// input.
//
// Usage:
//   1. Serve drishti-web:  npx serve drishti-web -l 5000
//   2. Launch headless Chrome with CDP:
//      chrome --headless=new --disable-gpu --remote-debugging-port=9333
//   3. node research/test_scenarios/run_scenarios.mjs
//
// Writes research/results/hazard_scenarios_<timestamp>.json and prints a
// summary table. Frame size is 1280x720 throughout, matching the resolution
// already used for the app's own real latency measurements (see
// PERFORMANCE_REPORT.md).

import { writeFileSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CDP = process.env.CDP_URL || 'http://localhost:9333';
const APP_URL = process.env.DRISHTI_URL || 'http://localhost:5000/';
const FRAME_W = 1280, FRAME_H = 720;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// [id, description, class, bbox:[x,y,w,h]] — see scenarios.md for the
// rationale behind each one's position/size.
const SCENARIOS = [
  [1, 'Person standing directly ahead, close range, in the walking corridor', 'person', [340, 150, 600, 550]],
  [2, 'Person far across the street, small in frame', 'person', [600, 300, 40, 80]],
  [3, 'Car approaching head-on in the corridor, close', 'car', [340, 150, 600, 550]],
  [4, 'Car parked at the far side of the frame, not in the walking corridor', 'car', [20, 600, 80, 60]],
  [5, 'Bicycle passing on the far left, not in the corridor', 'bicycle', [5, 200, 200, 460]],
  [6, 'Dog close by, directly ahead', 'dog', [400, 300, 500, 400]],
  [7, 'Chair directly ahead, close (indoor obstacle)', 'chair', [340, 150, 600, 550]],
  [8, 'Bench off to the side, far away', 'bench', [20, 600, 80, 60]],
  [9, 'Motorcycle close, off to the side but not in the direct corridor', 'motorcycle', [100, 150, 400, 500]],
  [10, 'Bus far away, centered', 'bus', [560, 300, 160, 120]],
  [11, 'Stop sign nearby, off to the side', 'stop sign', [1000, 300, 200, 150]],
  [12, 'Truck close, in the corridor, cut off by the bottom frame edge', 'truck', [400, 200, 650, 520]],
];

async function main() {
  const tab = await (await fetch(CDP + '/json/new?about:blank', { method: 'PUT' })).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r));
  let id = 1;
  function send(method, params = {}) {
    return new Promise(resolve => {
      const myId = id++;
      const h = (ev) => { const d = JSON.parse(ev.data); if (d.id === myId) { ws.removeEventListener('message', h); resolve(d.result); } };
      ws.addEventListener('message', h);
      ws.send(JSON.stringify({ id: myId, method, params }));
    });
  }
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: APP_URL });
  await sleep(1500);

  const results = [];
  for (const [num, description, cls, bbox] of SCENARIOS) {
    const expr = `
      (()=>{
        const obj = {class: ${JSON.stringify(cls)}, score: 0.9, bbox: ${JSON.stringify(bbox)}};
        const hazard = scoreHazard(obj, ${FRAME_W}, ${FRAME_H});
        const inPath = isInPathCorridor(obj.bbox, ${FRAME_W}, ${FRAME_H});
        const clock = bboxClockPosition(obj.bbox, ${FRAME_W});
        const prox = proximityFromBbox(obj.bbox, ${FRAME_W}, ${FRAME_H});
        return JSON.stringify({hazard, inPath, clock, proximity: prox});
      })()
    `;
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (r.exceptionDetails) {
      results.push({ id: num, description, class: cls, bbox, error: r.exceptionDetails.text });
      continue;
    }
    const parsed = JSON.parse(r.result.value);
    results.push({ id: num, description, class: cls, bbox, ...parsed });
  }

  ws.close();

  const outDir = join(__dirname, '..', 'results');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outFile = join(outDir, `hazard_scenarios_${stamp}.json`);
  writeFileSync(outFile, JSON.stringify({ frameW: FRAME_W, frameH: FRAME_H, generatedAt: new Date().toISOString(), results }, null, 2));

  console.log('\n# | Scenario | Class | Risk | Score | Proximity | In-path | Clock | Dist. confidence');
  console.log('--|----------|-------|------|-------|-----------|---------|-------|------------------');
  for (const r of results) {
    if (r.error) { console.log(`${r.id} | ${r.description} | ${r.class} | ERROR: ${r.error}`); continue; }
    const risk = r.hazard ? r.hazard.riskLevel : 'none (not a scored hazard)';
    const score = r.hazard ? r.hazard.score.toFixed(2) : '-';
    console.log(`${r.id} | ${r.description} | ${r.class} | ${risk} | ${score} | ${r.proximity.bucket} | ${r.inPath} | ${r.clock} | ${r.proximity.confidence}`);
  }
  console.log(`\nWrote ${results.length} real results to ${outFile}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
