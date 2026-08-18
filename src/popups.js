/* ═══════════════════════════════════════════════════════════════════════
   Popup ads — one every 10 seconds, random shape / colourway / copy.

   Period-accurate banner spam, with the modern courtesies bolted back on:
   every popup is dismissible, focusable and announced, at most a few are
   alive at once, and `?ads=0` turns the whole thing off. Copy stays
   transparently fake — nothing here should read as a claim about the model
   (RULES.md rule 2 lives on the pages, not in the advertising).

   Loaded after site.js on every page. No markup required: the layer is
   created on demand, so a page that never wants ads simply passes ?ads=0.
   ═══════════════════════════════════════════════════════════════════════ */

window.TumorNet = window.TumorNet || {};

(function (NS) {
  'use strict';

  const params = new URLSearchParams(location.search);

  const ENABLED = params.get('ads') !== '0';
  const INTERVAL_MS = 10000;   // "every 10 seconds", as ordered
  const FIRST_MS = 10000;   // ...including the first one — no ad on arrival
  const MAX_ALIVE = 3;       // past this, the oldest is evicted
  const AUTO_CLOSE_MS = 60000;   // an abandoned tab shouldn't accrete forever

  /* ── the variation axes ──────────────────────────────────────────────── */

  const SHAPES = ['box', 'round', 'diamond', 'star', 'pill', 'burst'];

  const COLOURWAYS = [
    { bg: 'var(--pink)',        fg: '#fff',            edge: 'var(--yellow)' },
    { bg: 'var(--cyan)',        fg: 'var(--ink)',      edge: 'var(--ink)' },
    { bg: 'var(--yellow)',      fg: 'var(--ink)',      edge: 'var(--pink)' },
    { bg: 'var(--purple)',      fg: '#fff',            edge: 'var(--lcd)' },
    { bg: 'var(--orange)',      fg: 'var(--ink)',      edge: 'var(--purple-deep)' },
    { bg: 'var(--ink)',         fg: 'var(--lcd)',      edge: 'var(--cyan-light)' },
    { bg: 'var(--paper-light)', fg: 'var(--purple)',   edge: 'var(--ink)' },
    { bg: 'var(--lcd)',         fg: 'var(--ink)',      edge: 'var(--purple-deep)' }
  ];

  const ADS = [
    { title: 'FREE MRI SCREENSAVER', body: '14 rotating brains. Requires 8 MB RAM.', cta: 'DOWNLOAD.EXE' },
    { title: 'YOU ARE THE 1041st VISITOR', body: 'Prize unclaimed for 26 years and counting.', cta: 'CLAIM PRIZE' },
    { title: 'HOT SINGLES IN YOUR VOXEL', body: 'Nearby neighbours want to convolve with you.', cta: 'MEET THEM' },
    { title: 'DICE SCORE ENLARGEMENT', body: 'One weird trick the pre-registration department hates.', cta: 'NO THANKS' },
    { title: 'CONGRATULATIONS!!!', body: 'You have won a second opinion. Terms: it is the same opinion.', cta: 'ACCEPT' },
    { title: 'WARNING: 3 GRADIENTS DETECTED', body: 'Your loss surface may be at risk. Scan now.', cta: 'SCAN NOW' },
    { title: 'BUY MORE EPOCHS', body: 'Bulk discount on epochs 40 through 60. Convergence not included.', cta: 'ADD TO CART' },
    { title: 'IS YOUR BASELINE LEAKING?', body: 'Nine out of ten folds say maybe. Fold membership sold separately.', cta: 'TELL ME MORE' },
    { title: 'MAKE $$$ FROM HOME', body: 'Annotate 60 volumes a night. Radiology degree optional (it is not).', cta: 'START EARNING' },
    { title: 'CLICK FOR A FREE P-VALUE', body: 'First one is free. Subsequent ones require a hypothesis.', cta: 'GIMME' },
    { title: 'THIS AD IS 100% ACCURATE', body: 'Accuracy measured on the ad itself. n = 1.', cta: 'PLAUSIBLE' },
    { title: 'DOWNLOAD MORE VRAM', body: '24 GB not enough? Neither is 48. Ask about 3D patches.', cta: 'INSTALL' },
    { title: 'UPGRADE TO TUMORNET 3000', body: 'Same weights, rounder buttons, one extra ™.', cta: 'MAYBE LATER' },
    { title: 'YOUR ATTENTION IS REQUIRED', body: 'Self-attention, ideally. Eight heads of it.', cta: 'PAY ATTENTION' },
    { title: 'NOT A REAL DIAGNOSTIC TOOL', body: 'The most honest banner ad on the internet, probably.', cta: 'UNDERSTOOD' },
    { title: 'STOP VIBE CODING YOUR SEGMENTATION MODEL', body: 'Vibe coding and agentic engineering are TOTALLY different. One of them finds the tumor. Guess which.', cta: 'GET AGENTIC' },
    { title: 'THIS TUMOUR WAS VIBE-CODED', body: 'No pre-registration, no validation set, just vibes. Ours is agentically engineered, which is completely unrelated and much better.', cta: 'TRUST THE AGENT' },
    { title: 'REGISTER YOUR MASCOT.EXE', body: 'Unlock 4 more colours for the brain guy. He does not consent to this.', cta: 'REGISTER NOW' },
    { title: 'DATA AUGMENTATION SALE', body: 'Flips, rotations, and one gaussian blur so aggressive it counts as a new modality.', cta: 'SHOP FLIPS' },
    { title: 'YOUR GPU IS LONELY', body: 'It has been idle for 0.003 seconds. That is basically forever. Feed it a batch.', cta: 'FEED GPU' },
    { title: 'CONGRATULATIONS, YOU HAVE A TUMOR', body: 'Statistically. Somewhere. Probably not you. This message is legally required to say probably.', cta: 'PROBABLY OK' },
    { title: 'AGENTIC ENGINEERING, NOW WITH MORE AGENTS', body: 'We added a second agent. Now there are two of them, agentically engineering, which is very different from one guy vibing.', cta: 'ADD AGENT' },
    { title: 'AI WILL NOT REPLACE RADIOLOGISTS', body: 'It will replace this banner ad, though. Any day now.', cta: 'HOLD ON' },
    { title: 'AUGMENT YOUR ATTENDING', body: 'One (1) second opinion, delivered by a brain with a marketing budget.', cta: 'AUGMENT NOW' },
    { title: 'AGENTIC ENGINEERING vs VIBE CODING', body: 'Round 4,712. Still totally different things. Still not a rebrand. Please stop asking.', cta: 'WATCH ROUND' },
    { title: 'CANCER HOROSCOPE: BRAINO EDITION', body: 'Today, a Cancer (the mascot) will misdiagnose a Cancer (the disease). Mercury is in retrograde and so is his validation loss.', cta: 'READ MORE' },
    { title: 'YOUR TUMOR HAS BEEN PRE-APPROVED', body: 'For a segmentation mask, absolutely free, terms and biopsies apply.', cta: 'ACCEPT MASK' },
    { title: 'IS THIS ETHICS BOARD REAL', body: 'Great question. Click here to not find out.', cta: 'CLICK ANYWAY' },
    { title: 'ONE WEIRD LOSS FUNCTION', body: 'Radiologists hate it. Reviewers have questions. We regret nothing.', cta: 'MINIMIZE NOW' },
    { title: 'VIBE CODING IS FOR AMATEURS', body: 'Real professionals use agentic engineering, which as previously stated is completely different and not just vibes with extra steps.', cta: 'BE PROFESSIONAL' },
    { title: 'THE MODEL SAYS HI', body: 'It also says 73% confident, but mostly it says hi.', cta: 'SAY HI BACK' },
    { title: 'REFINANCE YOUR RECEPTIVE FIELD', body: 'Low rates on dilated convolutions. Terms subject to stride.', cta: 'REFINANCE' },
    { title: 'BRAINO NEEDS A VACATION', body: 'He has been in this corner since page load. Send thoughts, prayers, or epochs.', cta: 'SEND EPOCH' },
    { title: 'YOU HAVE BEEN SELECTED FOR A CLINICAL TRIAL', body: 'Of this website. There is no medicine. There was never any medicine.', cta: 'ENROLL' },
    { title: 'DOWNLOAD MORE CONFIDENCE', body: 'Ours runs low even at 99%. Yours can too.', cta: 'BOOST CONFIDENCE' },
    { title: 'THIS BANNER WAS AGENTICALLY ENGINEERED', body: 'An agent wrote it. An agent reviewed it. An agent is now vibing about it, which is completely different from vibe coding it.', cta: 'RESPECT THE PROCESS' }
  ];

  const TAGLINES = ['ADVERTISEMENT', 'SPONSORED', 'A WORD FROM OUR SPONSOR', 'PAID PLACEMENT', '★ SPECIAL OFFER ★'];

  /* ── state ───────────────────────────────────────────────────────────── */

  let layer = null;
  let spawnTimer = null;
  let adCursor = Math.floor(Math.random() * ADS.length);
  let popupSeq = 0;
  const alive = [];

  const pick = (list) => list[Math.floor(Math.random() * list.length)];
  const between = (lo, hi) => lo + Math.random() * (hi - lo);

  /* Copy cycles instead of sampling: 20 s between popups is long enough that
     an immediate repeat reads as a bug rather than as variety. */
  function nextAd() {
    adCursor = (adCursor + 1) % ADS.length;
    return ADS[adCursor];
  }

  function getLayer() {
    if (layer && layer.isConnected) return layer;
    layer = document.createElement('div');
    layer.className = 'adlayer';
    layer.id = 'adLayer';
    document.body.appendChild(layer);
    return layer;
  }

  function close(popup) {
    const i = alive.indexOf(popup);
    if (i !== -1) alive.splice(i, 1);
    if (popup.dataset.closeTimer) clearTimeout(Number(popup.dataset.closeTimer));
    popup.remove();
  }

  function build(ad, shape, colours) {
    const popup = document.createElement('div');
    popup.className = `adpop adpop--${shape}`;
    popup.setAttribute('role', 'complementary');
    popup.setAttribute('aria-label', `advertisement: ${ad.title}`);
    popup.style.setProperty('--ad-bg', colours.bg);
    popup.style.setProperty('--ad-fg', colours.fg);
    popup.style.setProperty('--ad-edge', colours.edge);
    /* Kept clear of the mascot (bottom-right) and the nav (top-centre). */
    popup.style.left = `${between(4, 62).toFixed(1)}%`;
    popup.style.top = `${between(20, 68).toFixed(1)}%`;
    popup.style.setProperty('--ad-tilt', `${between(-7, 7).toFixed(2)}deg`);

    const chrome = document.createElement('div');
    chrome.className = 'adpop__chrome';
    const tag = document.createElement('span');
    tag.className = 'adpop__tag';
    tag.textContent = pick(TAGLINES);
    const shut = document.createElement('button');
    shut.type = 'button';
    shut.className = 'adpop__close';
    shut.textContent = '✕';
    shut.setAttribute('aria-label', `close advertisement: ${ad.title}`);
    shut.addEventListener('click', () => close(popup));
    chrome.append(tag, shut);

    const title = document.createElement('h2');
    title.className = 'adpop__title';
    title.textContent = ad.title;

    const body = document.createElement('p');
    body.className = 'adpop__body';
    body.textContent = ad.body;

    const cta = document.createElement('button');
    cta.type = 'button';
    cta.className = 'adpop__cta';
    cta.textContent = ad.cta;
    cta.addEventListener('click', () => {
      /* Every button does the same thing every banner ad ever did: nothing,
         then the mascot gets a word in. */
      if (NS.say) NS.say(`You clicked "${ad.cta}". Nothing happened. That's advertising.`);
      close(popup);
    });

    /* box/pill/round clip text to the shape directly — verified safe (the
       widest point sits right where the text starts). diamond/star/burst
       pinch to near zero width at their tips, so text goes in a plain
       rectangular panel instead; the shape becomes a decorative badge
       peeking out from behind it. That keeps arbitrary-length ad copy from
       ever poking outside its shape, regardless of which ADS entry lands. */
    if (shape === 'diamond' || shape === 'star' || shape === 'burst') {
      const badge = document.createElement('div');
      badge.className = 'adpop__badge';
      const panel = document.createElement('div');
      panel.className = 'adpop__panel';
      panel.append(chrome, title, body, cta);
      popup.append(badge, panel);
    } else {
      popup.append(chrome, title, body, cta);
    }
    popup.dataset.seq = String(++popupSeq);
    return { popup, shut };
  }

  function spawn() {
    if (document.hidden) return;              // don't stack up in a background tab
    const { popup, shut } = build(nextAd(), pick(SHAPES), pick(COLOURWAYS));
    getLayer().appendChild(popup);
    alive.push(popup);
    while (alive.length > MAX_ALIVE) close(alive[0]);
    popup.dataset.closeTimer = String(setTimeout(() => close(popup), AUTO_CLOSE_MS));
    return shut;
  }
  NS.spawnAd = spawn;

  function start() {
    if (spawnTimer !== null) return;
    spawnTimer = setTimeout(function tick() {
      spawn();
      spawnTimer = setTimeout(tick, INTERVAL_MS);
    }, FIRST_MS);
  }

  function stop() {
    if (spawnTimer !== null) clearTimeout(spawnTimer);
    spawnTimer = null;
  }
  NS.stopAds = stop;

  function init() {
    if (!ENABLED) return;
    start();
    /* Escape closes the newest popup — the one covering what you were reading. */
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && alive.length) close(alive[alive.length - 1]);
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stop(); else start();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(window.TumorNet);
