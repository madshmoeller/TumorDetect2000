/* ═══════════════════════════════════════════════════════════════════════
   TumorNet 2000 ™ — UI controller

   Owns DOM state only. All detection work goes through TumorNet.pipeline
   (see pipeline.js); this file never knows how a result was produced.
   ═══════════════════════════════════════════════════════════════════════ */

(function (NS) {
  'use strict';

  // This file is the scanner page only. Every other page includes site.js
  // for the shared chrome (nav, hit counter, guestbook, mascot) and skips
  // this script entirely. Guard here too, in case index.html's markup ever
  // changes: no case grid, no scanner, don't wire anything or throw.
  if (!document.getElementById('caseGrid')) return;

  // Config (mascotName / autoRunDetection / comparisonView) is parsed once
  // by site.js, which loads first on every page. Read it, don't recompute it.
  const config = NS.config || (NS.config = { mascotName: 'Braino', autoRunDetection: false, comparisonView: 'slider' });

  /* ── element lookup ──────────────────────────────────────────────────── */

  const $ = (id) => document.getElementById(id);

  const el = {
    caseGrid:       $('caseGrid'),
    emptyState:     $('emptyState'),
    casePanel:      $('casePanel'),
    loadedFile:     $('loadedFile'),
    detectBtn:      $('detectBtn'),

    analysisPanel:  $('analysisPanel'),
    analysisMsg:    $('analysisMsg'),
    progressBar:    $('progressBar'),
    progressFill:   $('progressFill'),
    progressPct:    $('progressPct'),

    previewPanel:   $('previewPanel'),
    previewImg:     $('previewImg'),

    resultsPanel:   $('resultsPanel'),
    predictionToggle:      $('predictionToggle'),
    predictionToggleState: $('predictionToggleState'),
    diffView:       $('diffView'),
    diffBase:       $('diffBase'),
    diffOverlay:    $('diffOverlay'),
    diffLocked:     $('diffLocked'),
    sliderView:     $('sliderView'),
    sliderTrack:    $('sliderTrack'),
    truthBase:      $('truthBase'),
    truthOverlay:   $('truthOverlay'),
    predReveal:     $('predReveal'),
    predBase:       $('predBase'),
    predOverlay:    $('predOverlay'),
    sliderLine:     $('sliderLine'),
    sliderHandle:   $('sliderHandle'),
    sliderLockedHint: $('sliderLockedHint'),

    statConfidence:    $('statConfidence'),
    statConfidenceBar: $('statConfidenceBar'),
    statVolume:        $('statVolume'),
    statVerdict:       $('statVerdict')
  };

  // Mascot bubble is site.js's — it owns the greeting, trivia and eye
  // tracking already; this page only needs to *say* things through it.
  const say = NS.say || function () {};

  /* ── state ───────────────────────────────────────────────────────────── */

  const state = {
    selectedId: null,
    analyzing: false,
    progress: 0,
    progressMessage: '',
    result: null,        // last Detection
    resultForId: null,   // which scan it belongs to
    predictionOn: false,  // model prediction is locked behind the paywall by default
    sliderPct: 50
  };

  let abortController = null;

  const scanById = (id) => NS.scans.find((s) => s.id === id) || null;

  /* Derived view phase, exactly as the prototype computed it: a result is
     only held for the most recent run, so switching scans returns to 'ready'. */
  function phase() {
    if (!state.selectedId) return 'empty';
    if (state.analyzing) return 'analyzing';
    if (state.result && state.resultForId === state.selectedId) return 'results';
    return 'ready';
  }

  /* ── formatting ──────────────────────────────────────────────────────── */

  // 0.76 -> "76%",  0.999 -> "99.9%"
  const formatConfidence = (v) => {
    const pct = v * 100;
    return (Math.round(pct * 10) / 10).toString() + '%';
  };

  const formatVolume = (mm3) => `${Math.round(mm3).toLocaleString('en-US')} mm³`;

  /* ── patient files ───────────────────────────────────────────────────── */

  function renderCaseGrid() {
    el.caseGrid.replaceChildren(...NS.scans.map((scan) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'case-card';
      card.dataset.scanId = scan.id;
      card.setAttribute('role', 'radio');
      card.setAttribute('aria-checked', 'false');

      const img = document.createElement('img');
      img.className = 'case-card__thumb';
      img.src = scan.base;
      img.alt = '';
      img.loading = 'lazy';

      const name = document.createElement('span');
      name.className = 'case-card__name';
      name.textContent = scan.filename;

      card.append(img, name);
      card.addEventListener('click', () => selectScan(scan.id));
      return card;
    }));
  }

  function selectScan(id) {
    cancelRun();
    state.selectedId = id;
    state.progress = 0;
    state.progressMessage = '';
    state.sliderPct = 50;
    render();
    say(NS.mascotLines.scanLoaded());
    if (config.autoRunDetection) setTimeout(runDetection, 250);
  }

  /* ── detection ───────────────────────────────────────────────────────── */

  function cancelRun() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    state.analyzing = false;
  }

  async function runDetection() {
    const scan = scanById(state.selectedId);
    if (!scan) {
      say(NS.mascotLines.noScan());
      return;
    }
    if (state.analyzing) return;

    cancelRun();
    abortController = new AbortController();
    const controller = abortController;

    state.analyzing = true;
    state.progress = 0;
    state.progressMessage = NS.progressMessages[0];
    // Model prediction is locked per result, unless the fake subscription is
    // already active this session — then it stays unlocked (see setPrediction).
    state.predictionOn = !!(NS.paywall && NS.paywall.isPaid());
    render();
    say(state.progressMessage);

    try {
      const detection = await NS.pipeline.detect(scan, {
        signal: controller.signal,
        onProgress: ({ pct, message }) => {
          if (controller.signal.aborted) return;
          state.progress = Math.max(0, Math.min(100, pct));
          if (message) state.progressMessage = message;
          renderProgress();
          say(state.progressMessage);
        }
      });

      if (controller.signal.aborted) return;

      state.analyzing = false;
      state.result = detection;
      state.resultForId = scan.id;
      state.sliderPct = 50;
      render();
      say(state.predictionOn
        ? detection.quip
        : ((NS.mascotLines && NS.mascotLines.paywall && NS.mascotLines.paywall()) || detection.quip));
    } catch (err) {
      if (err && err.name === 'AbortError') return;   // superseded by a new run
      console.error('[TumorNet] detection failed:', err);
      state.analyzing = false;
      state.result = null;
      state.resultForId = null;
      render();
      say(NS.mascotLines.failed());
    } finally {
      if (abortController === controller) abortController = null;
    }
  }

  /* ── model prediction toggle ─────────────────────────────────────────── */
  /* Off by default for every fresh result. Turning it on goes through the
     shared fake paywall (src/paywall.js); turning it off again is free. */

  function setPrediction(on) {
    state.predictionOn = on;
    render();
    if (on) say(state.result && state.result.quip);
  }

  function togglePrediction() {
    if (state.predictionOn) { setPrediction(false); return; }
    if (NS.paywall) NS.paywall.open(() => setPrediction(true));
    else setPrediction(true); // no paywall markup on this page — fail open
  }

  /* ── drag-to-compare ─────────────────────────────────────────────────── */

  function setSlider(pct) {
    state.sliderPct = Math.max(0, Math.min(100, pct));
    renderSlider();
  }

  const pctFromClientX = (clientX) => {
    const rect = el.sliderTrack.getBoundingClientRect();
    return ((clientX - rect.left) / rect.width) * 100;
  };

  function bindSlider() {
    el.sliderTrack.addEventListener('pointerdown', (e) => {
      // Clicking the track jumps to that spot, then drags with the pointer.
      el.sliderTrack.setPointerCapture(e.pointerId);
      setSlider(pctFromClientX(e.clientX));
      e.preventDefault();
    });

    el.sliderTrack.addEventListener('pointermove', (e) => {
      if (!el.sliderTrack.hasPointerCapture(e.pointerId)) return;
      setSlider(pctFromClientX(e.clientX));
    });

    // Keyboard equivalent — the prototype was drag-only.
    el.sliderHandle.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 10 : 2;
      const moves = { ArrowLeft: -step, ArrowRight: step, Home: -100, End: 100 };
      if (!(e.key in moves)) return;
      e.preventDefault();
      e.stopPropagation();
      setSlider(e.key === 'Home' ? 0 : e.key === 'End' ? 100 : state.sliderPct + moves[e.key]);
    });
  }

  /* ── rendering ───────────────────────────────────────────────────────── */

  function renderProgress() {
    const pct = Math.round(state.progress);
    el.analysisMsg.textContent = state.progressMessage;
    el.progressFill.style.width = `${pct}%`;
    el.progressPct.textContent = `${pct}%`;
    el.progressBar.setAttribute('aria-valuenow', String(pct));
  }

  function renderSlider() {
    const pct = Math.round(state.sliderPct);
    el.predReveal.style.clipPath = `inset(0 0 0 ${pct}%)`;
    el.sliderLine.style.left = `${pct}%`;
    el.sliderHandle.style.left = `${pct}%`;
    el.sliderHandle.setAttribute('aria-valuenow', String(pct));
  }

  function setLayer(img, src) {
    if (src) {
      img.src = src;
      img.hidden = false;
    } else {
      img.removeAttribute('src');
      img.hidden = true;
    }
  }

  function renderResults() {
    const { result, predictionOn } = state;
    const scan = scanById(state.selectedId);
    if (!result || !scan) return;

    el.predictionToggle.setAttribute('aria-checked', String(predictionOn));
    el.predictionToggleState.textContent = predictionOn ? 'ON' : 'OFF';

    // Honour the configured view, but only if the result carries what it needs.
    const useDiff = config.comparisonView === 'diff' && !!result.overlays.difference;

    el.diffView.hidden = !(useDiff && predictionOn);
    el.diffLocked.hidden = !(useDiff && !predictionOn);
    el.sliderView.hidden = useDiff;

    if (useDiff) {
      if (predictionOn) {
        setLayer(el.diffBase, scan.base);
        setLayer(el.diffOverlay, result.overlays.difference);
      }
    } else {
      setLayer(el.truthBase, scan.base);
      setLayer(el.truthOverlay, result.overlays.truth);
      el.predReveal.hidden = !predictionOn;
      el.sliderLine.hidden = !predictionOn;
      el.sliderHandle.hidden = !predictionOn;
      el.sliderLockedHint.hidden = predictionOn;
      if (predictionOn) {
        setLayer(el.predBase, scan.base);
        setLayer(el.predOverlay, result.overlays.prediction);
        renderSlider();
      }
    }

    if (predictionOn) {
      el.statConfidence.textContent = formatConfidence(result.confidence);
      el.statConfidenceBar.style.width = `${Math.max(0, Math.min(100, result.confidence * 100))}%`;
      el.statVolume.textContent = formatVolume(result.volumeMm3);
      el.statVerdict.textContent = result.verdict;
    } else {
      el.statConfidence.textContent = '🔒';
      el.statConfidenceBar.style.width = '0%';
      el.statVolume.textContent = '🔒';
      el.statVerdict.textContent = 'locked — see MODEL PREDICTION above';
    }
  }

  function render() {
    const p = phase();
    const scan = scanById(state.selectedId);

    el.caseGrid.querySelectorAll('.case-card').forEach((card) => {
      card.setAttribute('aria-checked', String(card.dataset.scanId === state.selectedId));
    });

    el.emptyState.hidden = p !== 'empty';
    el.casePanel.hidden = p === 'empty';
    if (p === 'empty') return;

    el.loadedFile.textContent = scan.filename;

    el.detectBtn.textContent =
      p === 'analyzing' ? 'ANALYZING...' : p === 'results' ? 'RUN AGAIN' : 'DETECT TUMOR!';
    el.detectBtn.disabled = p === 'analyzing';

    el.analysisPanel.hidden = p !== 'analyzing';
    el.previewPanel.hidden  = p !== 'ready';
    el.resultsPanel.hidden  = p !== 'results';

    if (p === 'analyzing') renderProgress();
    if (p === 'ready') {
      el.previewImg.src = scan.base;
      el.previewImg.alt = `Scan ${scan.filename}`;
    }
    if (p === 'results') renderResults();
  }

  /* ── boot ────────────────────────────────────────────────────────────── */

  function init() {
    // Mascot text swap, hit counter, guestbook and the greeting are site.js's
    // job and already ran (site.js loads before this file). This is just the
    // scanner: the case grid and the detect flow.
    renderCaseGrid();
    bindSlider();
    render();

    el.detectBtn.addEventListener('click', runDetection);
    el.predictionToggle.addEventListener('click', togglePrediction);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(window.TumorNet);
