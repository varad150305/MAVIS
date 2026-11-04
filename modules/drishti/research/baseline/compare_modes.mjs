// Real, runnable comparison between the three approaches research/README.md
// defines: Baseline A (generic prompt), Baseline B (Drishti's pre-migration,
// task-specific but cloud-only prompt), and Drishti Hybrid (the current
// local-first pipeline). Drives the real running app via Chrome DevTools
// Protocol and calls its actual, unmodified functions (_callGroqOnly,
// callLocalVLM, PROMPTS.describe) — never a reimplementation of the prompts
// or logic, per this project's own standing rule against fabricating results.
//
// WHAT THIS NEEDS TO PRODUCE REAL NUMBERS (deliberately not bundled/faked):
//   1. Real photos in research/datasets/images/ — indoor/outdoor scenes,
//      licensed for use, NOT synthetic/AI-generated (see research/README.md).
//   2. A real Groq or Gemini API key, for Baselines A and B specifically —
//      set GROQ_KEY or GEMINI_KEY as an environment variable before running.
//      Drishti Hybrid needs NO key (it's local-first), but does need the
//      on-device model already loaded once in the browser profile this
//      script drives (see README below for the two-step setup).
//
// Without real images, this script tells you that plainly and exits — it
// does not invent placeholder results.
//
// Usage:
//   GROQ_KEY=gsk_xxx node research/baseline/compare_modes.mjs
//   (Gemini instead: GEMINI_KEY=xxx node research/baseline/compare_modes.mjs)

import { readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join, extname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CDP = process.env.CDP_URL || 'http://localhost:9333';
const APP_URL = process.env.DRISHTI_URL || 'http://localhost:5000/';
const IMAGES_DIR = join(__dirname, '..', 'datasets', 'images');
const GENERIC_PROMPT = 'Describe what is in this image.';

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  if (!existsSync(IMAGES_DIR)) {
    console.log(`No dataset yet: ${IMAGES_DIR} does not exist.`);
    console.log('Add real photos there (see research/README.md) and re-run. Nothing to compare — exiting, not fabricating results.');
    return;
  }
  const files = readdirSync(IMAGES_DIR).filter(f => ['.jpg', '.jpeg', '.png', '.webp'].includes(extname(f).toLowerCase()));
  if (!files.length) {
    console.log(`No image files found in ${IMAGES_DIR}. Add real photos and re-run.`);
    return;
  }

  const groqKey = process.env.GROQ_KEY || '';
  const geminiKey = process.env.GEMINI_KEY || '';
  if (!groqKey && !geminiKey) {
    console.log('No GROQ_KEY or GEMINI_KEY environment variable set — Baselines A and B need a real cloud key (Drishti Hybrid does not, and will still run).');
  }

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

  if (groqKey) await send('Runtime.evaluate', { expression: `groqKey=${JSON.stringify(groqKey)};` });
  if (geminiKey) await send('Runtime.evaluate', { expression: `geminiKey=${JSON.stringify(geminiKey)};` });

  const results = [];
  for (const file of files) {
    const b64 = readFileSync(join(IMAGES_DIR, file)).toString('base64');
    console.log(`\n--- ${file} ---`);
    const row = { file };

    // Drishti Hybrid — the current, real local-first pipeline (mode='describe').
    // Uses whatever is actually configured (local model; cloud only as a
    // real fallback if the local model itself fails and a key is present —
    // see callGroq() in drishti-web/index.html).
    const hybridExpr = `
      (async()=>{
        mode='describe'; aiMode='LOCAL_ONLY';
        const t0=performance.now();
        try{
          const text=await callGroq(${JSON.stringify(b64)}, null, null);
          return JSON.stringify({ok:true, text, latencyMs:Math.round(performance.now()-t0)});
        }catch(e){ return JSON.stringify({ok:false, error:e.message, latencyMs:Math.round(performance.now()-t0)}); }
      })()
    `;
    const hybridRes = await send('Runtime.evaluate', { expression: hybridExpr, awaitPromise: true, returnByValue: true });
    row.hybrid = JSON.parse(hybridRes.result.value);
    console.log('Hybrid (local-first):', row.hybrid.ok ? `${row.hybrid.latencyMs}ms — "${row.hybrid.text.slice(0,120)}..."` : `FAILED — ${row.hybrid.error}`);

    if (groqKey || geminiKey) {
      // Baseline B — Drishti's own engineered, task-specific prompt, but
      // forced through the cloud (aiMode='CLOUD_ENHANCED' so _callGroqOnly
      // doesn't refuse), matching what the app looked like before this
      // migration (API_DEPENDENCY_REDUCTION.md's "before" column).
      const baselineBExpr = `
        (async()=>{
          mode='describe'; aiMode='CLOUD_ENHANCED';
          const t0=performance.now();
          try{
            const text=await callGroq(${JSON.stringify(b64)}, null, PROMPTS.describe);
            return JSON.stringify({ok:true, text, latencyMs:Math.round(performance.now()-t0)});
          }catch(e){ return JSON.stringify({ok:false, error:e.message, latencyMs:Math.round(performance.now()-t0)}); }
        })()
      `;
      const bRes = await send('Runtime.evaluate', { expression: baselineBExpr, awaitPromise: true, returnByValue: true });
      row.baselineB = JSON.parse(bRes.result.value);
      console.log('Baseline B (task-specific cloud prompt):', row.baselineB.ok ? `${row.baselineB.latencyMs}ms — "${row.baselineB.text.slice(0,120)}..."` : `FAILED — ${row.baselineB.error}`);

      // Baseline A — a generic, naive prompt through the same cloud path.
      const baselineAExpr = `
        (async()=>{
          mode='describe'; aiMode='CLOUD_ENHANCED';
          const t0=performance.now();
          try{
            const text=await callGroq(${JSON.stringify(b64)}, null, ${JSON.stringify(GENERIC_PROMPT)});
            return JSON.stringify({ok:true, text, latencyMs:Math.round(performance.now()-t0)});
          }catch(e){ return JSON.stringify({ok:false, error:e.message, latencyMs:Math.round(performance.now()-t0)}); }
        })()
      `;
      const aRes = await send('Runtime.evaluate', { expression: baselineAExpr, awaitPromise: true, returnByValue: true });
      row.baselineA = JSON.parse(aRes.result.value);
      console.log('Baseline A (generic prompt):', row.baselineA.ok ? `${row.baselineA.latencyMs}ms — "${row.baselineA.text.slice(0,120)}..."` : `FAILED — ${row.baselineA.error}`);
    } else {
      row.baselineB = { skipped: 'no cloud key configured' };
      row.baselineA = { skipped: 'no cloud key configured' };
    }

    results.push(row);
  }

  ws.close();

  const outDir = join(__dirname, '..', 'results');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outFile = join(outDir, `baseline_comparison_${stamp}.json`);
  writeFileSync(outFile, JSON.stringify({ generatedAt: new Date().toISOString(), imagesDir: IMAGES_DIR, results }, null, 2));
  console.log(`\nWrote real comparison results for ${results.length} image(s) to ${outFile}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
