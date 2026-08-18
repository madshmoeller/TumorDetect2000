/* ═══════════════════════════════════════════════════════════════════════
   Shared site chrome — nav, hit counter, guestbook, mascot.

   Every page includes this after scans.js (for NS.trivia / NS.mascotLines /
   NS.guestbookEntries) and gets: the nav bar, the visitor counter, the
   read-only guestbook, and the mascot (greeting, trivia on click, eye
   tracking). Page-specific behaviour (the scanner's case grid and detect
   flow) lives in app.js / viewer.js and is guarded to no-op on pages that
   don't have that markup — this file is guarded the same way in reverse:
   every element lookup is optional, so a page missing the sidebar (say, a
   future minimal page) just quietly skips that piece instead of throwing
   partway through init and taking the rest of the chrome down with it.
   ═══════════════════════════════════════════════════════════════════════ */

window.TumorNet = window.TumorNet || {};

(function (NS) {
  'use strict';

  /* ── configuration ───────────────────────────────────────────────────
     Query-string knobs, shared across every page: ?mascot=Cortex applies
     everywhere; ?autorun= and ?view= only mean anything on the scanner page
     but are parsed once here so app.js doesn't need its own copy.
  */
  const params = new URLSearchParams(location.search);

  NS.config = Object.assign({
    mascotName: 'Braino',
    autoRunDetection: false,
    comparisonView: 'slider'
  }, NS.config, {
    mascotName: params.get('mascot') || (NS.config && NS.config.mascotName) || 'Braino',
    autoRunDetection: params.has('autorun')
      ? params.get('autorun') !== '0'
      : !!(NS.config && NS.config.autoRunDetection),
    comparisonView: (params.get('view') || (NS.config && NS.config.comparisonView) || 'slider')
      .toLowerCase().startsWith('diff') ? 'diff' : 'slider'
  });
  const config = NS.config;

  const PAGES = [
    { id: 'index',     href: 'index.html',     label: 'THE SCANNER' },
    { id: 'about',     href: 'about.html',     label: 'ABOUT US' },
    { id: 'results',   href: 'results.html',   label: 'THE NUMBERS' },
    { id: 'methods',   href: 'methods.html',   label: 'HOW IT WORKS' },
    { id: 'guestbook', href: 'guestbook.html', label: 'GUESTBOOK' }
  ];
  NS.pages = PAGES;

  const HITS_KEY = 'tumornet2000_hits';
  const GUESTBOOK_KEY = 'tumornet2000_guestbook';

  //: Server-side guestbook (netlify/functions/guestbook.mjs, backed by Netlify
  //: Blobs). localStorage alone meant "signing" saved nothing anybody else could
  //: see — every visitor kept a private copy. The endpoint is the source of truth
  //: when it answers; localStorage stays as the offline fallback so `python
  //: serve.py` and file:// keep working, where no function exists to call.
  const GUESTBOOK_API = '/api/guestbook';
  let remoteEntries = null;   // null = endpoint not reachable / not tried yet
  let signing = false;
  const HITS_SEED = 1041;
  const BUBBLE_MS = 10000;
  const EYE_RANGE = 140;
  const EYE_TRAVEL = 6;

  const SIDEBAR_ENTRY_COUNT = 5;

  const $ = (id) => document.getElementById(id);
  const el = {
    siteNav:           $('siteNav'),
    hitCounter:        $('hitCounter'),
    guestbookList:     $('guestbookList'),
    guestbookFullList: $('guestbookFullList'),
    guestbookName:     $('guestbookName'),
    guestbookMessage:  $('guestbookMessage'),
    guestbookSign:     $('guestbookSign'),
    guestbookWebsite:  $('guestbookWebsite'),
    mascotBubble:      $('mascotBubble'),
    brain:             $('brain')
  };

  let bubbleTimer = null;
  let triviaIndex = -1;
  let eyeFrame = null;

  /* ── nav ─────────────────────────────────────────────────────────────── */

  function renderNav() {
    if (!el.siteNav) return;
    const current = document.body.dataset.page || '';
    el.siteNav.replaceChildren(...PAGES.map((p) => {
      const a = document.createElement('a');
      a.href = p.href;
      a.className = 'sitenav__link';
      a.textContent = p.label;
      if (p.id === current) {
        a.setAttribute('aria-current', 'page');
        a.classList.add('sitenav__link--current');
      }
      return a;
    }));
  }

  /* ── mascot ──────────────────────────────────────────────────────────── */

  function say(text) {
    if (!text || !el.mascotBubble) return;
    if (bubbleTimer) clearTimeout(bubbleTimer);
    el.mascotBubble.textContent = text;
    el.mascotBubble.hidden = false;
    bubbleTimer = setTimeout(() => { el.mascotBubble.hidden = true; }, BUBBLE_MS);
  }
  NS.say = say;

  function cycleTrivia() {
    if (!NS.trivia || !NS.trivia.length) return;
    triviaIndex = (triviaIndex + 1) % NS.trivia.length;
    say(NS.trivia[triviaIndex](config.mascotName));
  }

  function trackEyes(event) {
    if (eyeFrame !== null || !el.brain) return;
    eyeFrame = requestAnimationFrame(() => {
      eyeFrame = null;
      const rect = el.brain.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height * 0.4;
      const dx = event.clientX - cx;
      const dy = event.clientY - cy;
      const dist = Math.hypot(dx, dy) || 1;
      const reach = Math.min(dist, EYE_RANGE) / EYE_RANGE * EYE_TRAVEL;
      const offset = `translate(${(dx / dist) * reach}px, ${(dy / dist) * reach}px)`;
      el.brain.querySelectorAll('[data-pupil]').forEach((p) => { p.style.transform = offset; });
    });
  }

  async function signGuestbook() {
    if (!el.guestbookName || signing) return;
    const name = el.guestbookName.value;
    const message = el.guestbookMessage ? el.guestbookMessage.value : '';
    if (!name || !name.trim()) {
      say('A name first. Anonymity is fine but blankness is rude.');
      return;
    }
    if (!message || !message.trim()) {
      say('Words too, please. I judge silently but I still need something to judge.');
      return;
    }

    signing = true;
    if (el.guestbookSign) el.guestbookSign.disabled = true;
    try {
      const status = await addGuestbookEntry(name, message.trim());
      if (status === 'throttled') {
        say('One signature at a time. Even I need a moment between judgements.');
        return;
      }
      if (status === 'invalid') return;
      el.guestbookName.value = '';
      if (el.guestbookMessage) el.guestbookMessage.value = '';
      say(status === 'shared'
        ? `${name} signed the guestbook, permanently, where everyone can see it. Bold.`
        : `${name} signed the guestbook — but only on this device. No backend here.`);
    } finally {
      signing = false;
      if (el.guestbookSign) el.guestbookSign.disabled = false;
    }
  }

  /* ── sidebar furniture ───────────────────────────────────────────────── */

  function renderHitCounter() {
    if (!el.hitCounter) return;
    let hits;
    try {
      hits = parseInt(localStorage.getItem(HITS_KEY) || String(HITS_SEED), 10) + 1;
      localStorage.setItem(HITS_KEY, String(hits));
    } catch (err) {
      hits = HITS_SEED + 1; // private browsing, file://, storage blocked, ...
    }
    const digits = String(hits).padStart(6, '0');
    el.hitCounter.replaceChildren(...[...digits].map((ch) => {
      const d = document.createElement('span');
      d.className = 'counter__digit';
      d.textContent = ch;
      return d;
    }));
    el.hitCounter.setAttribute('aria-label', `visitor number ${hits}`);
  }

  function getGuestbookEntries() {
    const defaultEntries = NS.guestbookEntries || [];
    // When the endpoint has answered, it is authoritative: showing the server's
    // list alongside a local copy would double up entries this browser signed.
    if (remoteEntries) return [...defaultEntries, ...remoteEntries];
    let stored = [];
    try {
      const data = localStorage.getItem(GUESTBOOK_KEY);
      stored = data ? JSON.parse(data) : [];
    } catch (err) {
      // private browsing, file://, storage blocked, ...
    }
    return [...defaultEntries, ...stored];
  }

  /** Pull the shared scroll. Silent on failure: a static local copy with no
   *  function backend is a supported way to run this site, not an error. */
  async function loadRemoteGuestbook() {
    if (!el.guestbookList && !el.guestbookFullList) return;
    try {
      const res = await fetch(GUESTBOOK_API, { cache: 'no-store' });
      if (!res.ok) return;
      const data = await res.json();
      if (!data || !Array.isArray(data.entries)) return;
      remoteEntries = data.entries;
      renderGuestbook();
      renderGuestbookFull();
    } catch (err) {
      // offline, no function, blocked — keep whatever localStorage gave us
    }
  }

  function addGuestbookEntryLocal(name, text) {
    const entry = { name: name.trim(), text };
    try {
      const data = localStorage.getItem(GUESTBOOK_KEY);
      const entries = data ? JSON.parse(data) : [];
      entries.push(entry);
      localStorage.setItem(GUESTBOOK_KEY, JSON.stringify(entries));
    } catch (err) {
      // private browsing, file://, storage blocked, ... can't persist
    }
  }

  /** Try the shared guestbook first, fall back to this browser only.
   *  Returns a status so the mascot can tell the truth about what happened —
   *  "signed" and "saved only on this device" are different outcomes and the
   *  user deserves to know which they got. */
  async function addGuestbookEntry(name, text) {
    if (!name || !name.trim()) return 'invalid';
    let status = 'local';
    try {
      const res = await fetch(GUESTBOOK_API, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        // The honeypot is sent as-is: the server treats any non-empty value as
        // a bot and quietly accepts-without-storing, so a filled field must
        // reach it rather than being stripped here.
        body: JSON.stringify({
          name: name.trim(),
          text,
          website: el.guestbookWebsite ? el.guestbookWebsite.value : ''
        })
      });
      if (res.status === 429) return 'throttled';
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.entries)) {
          remoteEntries = data.entries;
          status = 'shared';
        }
      }
    } catch (err) {
      // no endpoint (local static serve) — fall through to localStorage
    }
    if (status !== 'shared') addGuestbookEntryLocal(name, text);
    renderGuestbook();
    renderGuestbookFull();
    return status;
  }

  function buildGuestbookList(target, entries) {
    if (!target) return;
    target.replaceChildren(...entries.map((entry) => {
      const li = document.createElement('li');
      li.className = 'guestbook__entry';
      const who = document.createElement('span');
      who.className = 'guestbook__author';
      who.textContent = `${entry.name}: `;
      li.append(who, document.createTextNode(entry.text));
      return li;
    }));
  }

  function renderGuestbook() {
    if (!el.guestbookList) return;
    const entries = getGuestbookEntries();
    const mostRecentFirst = entries.slice(-SIDEBAR_ENTRY_COUNT).reverse();
    buildGuestbookList(el.guestbookList, mostRecentFirst);
  }

  function renderGuestbookFull() {
    if (!el.guestbookFullList) return;
    const entries = getGuestbookEntries();
    buildGuestbookList(el.guestbookFullList, entries.slice().reverse());
  }

  /* ── boot ────────────────────────────────────────────────────────────── */

  function init() {
    document.querySelectorAll('[data-mascot-name]').forEach((node) => {
      node.textContent = config.mascotName.toUpperCase();
    });

    renderNav();
    renderHitCounter();
    renderGuestbook();
    renderGuestbookFull();
    // Render the seeded/local list first so the page is never briefly empty,
    // then swap in the shared scroll when the endpoint answers.
    loadRemoteGuestbook();

    if (el.brain) el.brain.addEventListener('click', cycleTrivia);
    if (el.guestbookSign) {
      el.guestbookSign.addEventListener('click', signGuestbook);
    }
    if (el.guestbookName) {
      el.guestbookName.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') signGuestbook();
      });
    }

    window.addEventListener('mousemove', trackEyes);
    if (NS.mascotLines) say(NS.mascotLines.greeting(config.mascotName));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(window.TumorNet);
