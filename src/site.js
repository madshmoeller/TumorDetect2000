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

  function signGuestbook() {
    if (!el.guestbookName) return;
    const name = el.guestbookName.value;
    const message = el.guestbookMessage ? el.guestbookMessage.value : '';
    if (!message || !message.trim()) {
      say('Words too, please. I judge silently but I still need something to judge.');
      return;
    }
    if (addGuestbookEntry(name, message.trim())) {
      el.guestbookName.value = '';
      if (el.guestbookMessage) el.guestbookMessage.value = '';
      say(`${name} signed the guestbook. How thrilling.`);
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
    let stored = [];
    try {
      const data = localStorage.getItem(GUESTBOOK_KEY);
      stored = data ? JSON.parse(data) : [];
    } catch (err) {
      // private browsing, file://, storage blocked, ...
    }
    return [...defaultEntries, ...stored];
  }

  function addGuestbookEntry(name, text) {
    if (!name || !name.trim()) return false;
    const entry = { name: name.trim(), text };
    let entries = [];
    try {
      const data = localStorage.getItem(GUESTBOOK_KEY);
      entries = data ? JSON.parse(data) : [];
      entries.push(entry);
      localStorage.setItem(GUESTBOOK_KEY, JSON.stringify(entries));
    } catch (err) {
      // private browsing, file://, storage blocked, ... can't persist
    }
    renderGuestbook();
    renderGuestbookFull();
    return true;
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
