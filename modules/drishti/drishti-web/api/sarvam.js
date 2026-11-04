// Optional serverless proxy for Sarvam AI's Indian-language text-to-speech.
// Same reasoning as api/groq.js — set SARVAM_API_KEY once in your hosting
// provider's environment variables and every visitor gets natural Indian-voice
// speech with zero setup. If this isn't configured, Drishti automatically
// falls back to the browser's built-in speech (still works, just a more
// robotic voice), so this is a nice-to-have, not a requirement.
//
// Same rate-limiting rationale as api/groq.js — caps each visitor to 40
// requests/minute so one bot or bad actor can't burn the shared key for
// everyone else. See the longer comment in api/groq.js for the tradeoffs.
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

  if (req.method === 'GET') {
    return res.status(200).json({ ok: true, configured: !!process.env.SARVAM_API_KEY });
  }

  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  if (_isRateLimited(req)) return res.status(429).json({ error: 'rate_limited' });

  const key = process.env.SARVAM_API_KEY;
  if (!key) return res.status(500).json({ error: 'proxy_not_configured' });

  try {
    const sarvamRes = await fetch('https://api.sarvam.ai/text-to-speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'api-subscription-key': key },
      body: JSON.stringify(req.body)
    });
    const data = await sarvamRes.json();
    return res.status(sarvamRes.status).json(data);
  } catch (e) {
    return res.status(502).json({ error: 'upstream_unreachable' });
  }
}
