/* Guestbook persistence — Netlify Function + Blobs.
 *
 * The guestbook used to write to localStorage, which meant "signing" it saved
 * nothing anyone else could see: every visitor got their own private copy. This
 * stores entries server-side so the scroll is genuinely shared.
 *
 *   GET  /api/guestbook  -> { entries: [{name, text, at}], count }
 *   POST /api/guestbook  -> { ok, entry, entries, count }   body: {name, text}
 *
 * Netlify Forms was the other candidate and is simpler, but it has no public read
 * API — submissions land in a dashboard, so the page could never display them.
 * A guestbook nobody can read back is not a guestbook.
 */
import { getStore } from '@netlify/blobs';

const KEY = 'entries';
const MAX_NAME = 40;
const MAX_TEXT = 280;
const MAX_ENTRIES = 500; // oldest dropped past this
const MIN_MS_BETWEEN = 5000; // per-IP throttle
const CAS_RETRIES = 4;

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
  });

/** Strip control characters, collapse whitespace, hard-truncate.
 *
 *  The client renders with textContent, so markup cannot execute — but a
 *  20,000-character entry or a run of newlines would still wreck the layout for
 *  every future visitor. That is a storage-side concern, not a rendering one, so
 *  it is enforced here where it cannot be bypassed by a hand-rolled request.
 */
function clean(value, max) {
  return String(value ?? '')
    .replace(/[\u0000-\u001f\u007f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, max);
}

export default async (req, context) => {
  let store;
  try {
    store = getStore({ name: 'guestbook', consistency: 'strong' });
  } catch (err) {
    return json({ error: 'blob store unavailable', detail: String(err) }, 503);
  }

  if (req.method === 'GET') {
    const entries = (await store.get(KEY, { type: 'json' })) || [];
    return json({ entries, count: entries.length });
  }

  if (req.method !== 'POST') {
    return json({ error: 'method not allowed' }, 405);
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return json({ error: 'body must be JSON' }, 400);
  }

  // Honeypot: a field no human ever sees. Bots fill every input they find.
  if (clean(body.website, 10)) return json({ ok: true, ignored: true });

  const name = clean(body.name, MAX_NAME);
  const text = clean(body.text, MAX_TEXT);
  if (!name) return json({ error: 'name required' }, 400);
  if (!text) return json({ error: 'message required' }, 400);

  // Throttle by IP. Best-effort: it lives in a separate blob from the entries, so
  // it is advisory rather than a guarantee. An unmoderated public write endpoint
  // on a public site will still collect junk eventually — this only raises the
  // effort required.
  const ip = req.headers.get('x-nf-client-connection-ip') || context?.ip || 'unknown';
  const now = Date.now();
  try {
    const throttle = getStore({ name: 'guestbook-throttle', consistency: 'strong' });
    const last = Number((await throttle.get(`ip:${ip}`)) || 0);
    if (now - last < MIN_MS_BETWEEN) {
      return json({ error: 'slow down — one signature at a time' }, 429);
    }
    await throttle.set(`ip:${ip}`, String(now));
  } catch {
    // Throttling is a nicety; never fail a legitimate signature over it.
  }

  const entry = { name, text, at: new Date(now).toISOString() };

  // Compare-and-swap. A plain read-modify-write silently loses an entry whenever
  // two people sign in the same moment; retrying against the etag makes
  // concurrent signatures correct rather than merely unlikely.
  for (let attempt = 0; attempt <= CAS_RETRIES; attempt++) {
    const current = await store.getWithMetadata(KEY, { type: 'json' }).catch(() => null);
    const entries = (current && current.data) || [];
    const next = [...entries, entry].slice(-MAX_ENTRIES);
    try {
      const opts = current && current.etag ? { onlyIfMatch: current.etag } : { onlyIfNew: true };
      const res = await store.setJSON(KEY, next, opts);
      // Netlify reports {modified:false} when the precondition failed.
      if (res && res.modified === false) continue;
      return json({ ok: true, entry, entries: next, count: next.length });
    } catch (err) {
      // An older @netlify/blobs without conditional writes: fall back to a plain
      // write on the last attempt rather than failing the request outright.
      if (attempt === CAS_RETRIES) {
        await store.setJSON(KEY, next);
        return json({ ok: true, entry, entries: next, count: next.length, cas: false });
      }
    }
  }
  return json({ error: 'could not save, please retry' }, 503);
};

export const config = { path: '/api/guestbook' };
