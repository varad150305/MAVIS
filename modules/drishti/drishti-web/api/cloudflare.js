// Serverless CORS-bypass proxy for Cloudflare Workers AI.
//
// WHY THIS EXISTS: unlike Groq and Gemini, Cloudflare's REST API
// (api.cloudflare.com) sends no CORS headers, so a browser fetch() straight
// to it is blocked outright by the browser itself, regardless of whose
// account/token is used — this surfaced as a real, misleading "No internet
// connection reaching Cloudflare" error report even though the same session
// had just gotten real responses from Gemini and Groq (see
// IMPLEMENTATION_PROGRESS.md). A server-to-server call isn't subject to CORS,
// so relaying through here and attaching the CORS headers the browser needs
// on the way back is the actual fix — not something client-side code alone
// can work around.
//
// Unlike groq.js/sarvam.js, this proxy carries no secret of its own by
// default: if the site owner hasn't configured CLOUDFLARE_ACCOUNT_ID /
// CLOUDFLARE_API_TOKEN (Vercel env vars, same pattern as GROQ_API_KEY), it
// simply relays whatever account ID + token the caller supplies via headers
// — the exact credentials a visitor already pastes into the app's own
// Settings panel. Configuring the env vars instead gives every visitor a
// zero-setup Cloudflare fallback the same way GROQ_API_KEY does for Groq.
//
// Only reachable when deployed somewhere that runs serverless functions
// (Vercel, as this repo is set up for) — plain static hosting (GitHub Pages,
// Netlify drop, `npx serve`, or opening index.html directly) has no way to
// run this file, so Cloudflare's fallback simply isn't reachable there; the
// app's own client-side error message says so rather than mysteriously
// failing.
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
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-CF-Account, X-CF-Token, X-CF-Model');
  if (req.method === 'OPTIONS') return res.status(200).end();

  // Lightweight health check — the client calls this on page load, same
  // pattern as api/groq.js, to know whether a site-owner-configured shared
  // credential exists (purely informational here; unlike Groq, the app
  // doesn't hide any UI based on this since Cloudflare stays optional either way).
  if (req.method === 'GET') {
    return res.status(200).json({ ok: true, configured: !!(process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN) });
  }

  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  if (_isRateLimited(req)) return res.status(429).json({ error: 'rate_limited' });

  // Site owner's shared credentials (zero-setup for visitors) take priority
  // if configured; otherwise fall back to whatever the caller supplied
  // themselves. This proxy exists to bypass Cloudflare's lack of CORS
  // support either way, not to gate on whose credentials are used.
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID || req.headers['x-cf-account'];
  const token = process.env.CLOUDFLARE_API_TOKEN || req.headers['x-cf-token'];
  const model = req.headers['x-cf-model'];
  if (!accountId || !token) return res.status(400).json({ error: 'missing_credentials' });
  if (!model) return res.status(400).json({ error: 'missing_model' });

  try {
    const cfRes = await fetch(`https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${model}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
      body: JSON.stringify(req.body)
    });
    const data = await cfRes.json();
    return res.status(cfRes.status).json(data);
  } catch (e) {
    return res.status(502).json({ error: 'upstream_unreachable' });
  }
}
