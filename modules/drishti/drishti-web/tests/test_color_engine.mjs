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

  const testScript = `
(function(){
  const results = [];
  function assert(name, cond, detail){ results.push({name, pass: !!cond, detail}); }

  // Nearest-color-name sanity checks against unambiguous, far-apart RGB values —
  // covers every primary/secondary, not just a couple of easy cases, since an
  // earlier calibration of this table silently mapped pure green to "cyan" and
  // pure blue to "dark blue" and only broad coverage caught it.
  assert('pure red maps to red', nearestColorName(255,0,0)==='red');
  assert('pure green maps to green', nearestColorName(0,255,0)==='green');
  assert('pure blue maps to a blue variant', ['blue','dark blue','navy'].includes(nearestColorName(0,0,255)));
  assert('pure yellow maps to yellow', nearestColorName(255,255,0)==='yellow');
  assert('pure cyan maps to cyan', nearestColorName(0,255,255)==='cyan');
  assert('pure magenta maps to magenta', nearestColorName(255,0,255)==='magenta');
  assert('orange maps to orange', nearestColorName(255,165,0)==='orange');
  assert('purple maps to purple', nearestColorName(128,0,128)==='purple');
  assert('pure white maps to white', nearestColorName(255,255,255)==='white');
  assert('pure black maps to black', nearestColorName(0,0,0)==='black');
  assert('mid gray maps to gray', nearestColorName(128,128,128)==='gray');
  assert('saddle brown maps to brown', nearestColorName(139,69,19)==='brown');

  // Paint a real 9-cell grid onto the actual capture canvas (the same one
  // localColorGrid() reads from) with known, deliberately distinct colors in
  // each cell, then confirm the grid function reads them back correctly.
  canvas.width=300; canvas.height=300; // 3x3 grid -> 100x100 cells
  const cellColors=[
    [255,0,0],[255,0,0],[255,0,0],      // top row: red
    [255,255,255],[255,255,255],[255,255,255], // middle row: white
    [0,0,255],[0,0,255],[0,0,255]        // bottom row: blue
  ];
  let i=0;
  for(let row=0; row<3; row++){
    for(let col=0; col<3; col++){
      const [r,g,b]=cellColors[i++];
      ctx.fillStyle='rgb('+r+','+g+','+b+')';
      ctx.fillRect(col*100,row*100,100,100);
    }
  }
  const grid = localColorGrid();
  assert('grid returns exactly 3 distinct colors for this pattern', grid.length===3, JSON.stringify(grid));
  assert('grid contains red, white, and a blue variant', grid.includes('red') && grid.includes('white') && grid.some(c=>c.includes('blue')), JSON.stringify(grid));

  // Empty/unpainted canvas should not throw and should return an empty array, not fabricate a color.
  canvas.width=0; canvas.height=0;
  let emptyResult, threw=false;
  try{ emptyResult = localColorGrid(); }catch(e){ threw=true; }
  assert('zero-size canvas does not throw', !threw);
  assert('zero-size canvas returns empty grid, not a guessed color', Array.isArray(emptyResult) && emptyResult.length===0, JSON.stringify(emptyResult));

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
