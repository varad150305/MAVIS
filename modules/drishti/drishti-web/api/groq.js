// Serverless proxy for the Groq vision API.
//
// WHY THIS EXISTS: Drishti is built for blind and elderly users. Asking them to
// create a Groq account, generate an API key, and paste it into a password field
// is not a realistic first step for that audience — most would need a sighted
// helper just to get the app working at all. This proxy removes that barrier:
// the site owner (you) sets ONE environment variable once, and every visitor
// gets a working app with zero setup.
//
// Deploy on Vercel: Project Settings → Environment Variables → add
//   GROQ_API_KEY = <your free key from console.groq.com>
// Then redeploy. That's the entire setup, done once, for everyone who ever uses
// the site. Users who visit a copy of this site that ISN'T configured this way
// still see the manual "paste your own key" fallback — the app works either way.
//
// RATE LIMITING: since this proxy has no login and shares one key across every
// visitor, anyone who discovers the URL could otherwise hammer it and burn
// through your free-tier quota. The in-memory limiter below caps each visitor
// (identified by IP) to a generous 40 requests/minute — enough for one real
// person using every feature including Guardian Mode, but not a bot script.
// It resets if the function cold-starts, and isn't perfectly accurate across
// Vercel's multiple instances — that's an acceptable tradeoff for a zero-cost
// social-good project, not a bank. Swap in Vercel's KV/Upstash-backed rate
// limiting later if this ever needs to be airtight.
const _rateBuckets = new Map();
function _isRateLimited(req, windowMs = 60000, max = 40) {
  const ip = (req.headers['x-forwarded-for'] || req.socket?.remoteAddress || 'unknown').split(',')[0].trim();
  const now = Date.now();
  let bucket = _rateBuckets.get(ip);
  if (!bucket || now - bucket.start > windowMs) {
    bucket = { start: now, count: 0 };
    _rateBuckets.set(ip, bucket);
  }
  bucket.count++;
  if (_rateBuckets.size > 5000) {
    for (const [k, v] of _rateBuckets) if (now - v.start > windowMs) _rateBuckets.delete(k);
  }
  return bucket.count > max;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();

  // Lightweight health check — the client calls this on page load to decide
  // whether to hide the "paste your API key" screen entirely.
  if (req.method === 'GET') {
    return res.status(200).json({ ok: true, configured: !!process.env.GROQ_API_KEY });
  }

  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  if (_isRateLimited(req)) return res.status(429).json({ error: 'rate_limited' });

  const key = process.env.GROQ_API_KEY;
  if (!key) return res.status(500).json({ error: 'proxy_not_configured' });

  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
      body: JSON.stringify(req.body)
    });
    const data = await groqRes.json();
    return res.status(groqRes.status).json(data);
  } catch (e) {
    return res.status(502).json({ error: 'upstream_unreachable' });
  }
}
