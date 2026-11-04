// Proper automated TDZ checker: finds every top-level `let`/`const` name and its
// declaration line, then scans every top-level executable statement (function
// calls, assignments — not function/class declarations, which are hoisted) for
// references to those names that occur BEFORE the declaration line. This is the
// exact bug class that broke this app three separate times this session
// (vcOn, snapHistory, aiMode) — each time invisible to a human eyeballing a
// list of statements, since the declaration was hundreds of lines away.
import fs from 'fs';

const html = fs.readFileSync(process.argv[2] || 'index.html', 'utf8');
const startTag = html.indexOf('<script>');
const endTag = html.indexOf('</script>');
const js = html.slice(startTag + '<script>'.length, endTag);
const lines = js.split('\n');

// Pass 1: find all top-level let/const declarations and their line numbers.
const declLine = {};
let depth = 0;
lines.forEach((line, i) => {
  const opens = (line.match(/{/g) || []).length;
  const closes = (line.match(/}/g) || []).length;
  if (depth === 0) {
    const m = line.match(/^(?:let|const)\s+(.+?);?\s*$/);
    if (m) {
      // split on commas at the top level of the declaration (ignore commas inside strings/calls — good enough for this codebase's simple declarations)
      m[1].split(',').forEach(part => {
        const name = part.trim().split('=')[0].trim();
        if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(name) && !(name in declLine)) {
          declLine[name] = i + 1;
        }
      });
    }
  }
  depth += opens - closes;
});

// Pass 2: find top-level executable statements (not function/class/let/const
// declarations — those are hoisted or don't execute immediately) and check
// each for references to a name whose declaration comes later.
depth = 0;
let issues = 0;
lines.forEach((line, i) => {
  const opens = (line.match(/{/g) || []).length;
  const closes = (line.match(/}/g) || []).length;
  const lineNo = i + 1;
  if (depth === 0 && /^[a-zA-Z_$(]/.test(line) && !/^(function|async function|let|const|class)\s/.test(line)) {
    // crude identifier scan — good enough given this file's style (no minification)
    const idents = line.match(/\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g) || [];
    const seen = new Set();
    idents.forEach(name => {
      if (seen.has(name)) return;
      seen.add(name);
      if (declLine[name] && declLine[name] > lineNo) {
        console.log(`TDZ RISK: line ${lineNo} uses "${name}", but it isn't declared until line ${declLine[name]}`);
        console.log(`  -> ${line.trim()}`);
        issues++;
      }
    });
  }
  depth += opens - closes;
});

if (issues === 0) {
  console.log('No TDZ risks found — every top-level statement only references names already declared by that point.');
  process.exit(0);
} else {
  console.log(`\n${issues} potential TDZ issue(s) found.`);
  process.exit(1);
}
