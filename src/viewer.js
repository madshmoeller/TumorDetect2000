/* ═══════════════════════════════════════════════════════════════════════
   The real-data viewer — canvas-based MRI slice browser.

   Loads assets/cases/manifest.json (written by `python -m ml.export_web`).
   If it's missing or empty, this script does nothing and the legacy mock
   demo that app.js already rendered stays exactly as it was — that is the
   documented fallback, not an error path.

   Why canvas, not <img> layers like the legacy viewer: an atlas PNG holds
   every sampled slice for one case/channel. It is fetched once and decoded
   into an Image, and every slice change after that is a single drawImage
   blit from the atlas — no network request, no flash, real scrubbing.
   ═══════════════════════════════════════════════════════════════════════ */

window.TumorNet = window.TumorNet || {};

(function (NS) {
  'use strict';

  const legacyLab = document.getElementById('labLegacy');
  const realLab = document.getElementById('labReal');
  if (!realLab) return; // this page has no real viewer markup at all

  const $ = (id) => document.getElementById(id);
  const el = {
    caseGrid:          $('realCaseGrid'),
    splitSwitch:       $('splitSwitch'),
    scannerBlurb:      $('scannerBlurb'),
    emptyState:        $('realEmptyState'),
    panel:             $('realPanel'),
    loadedCase:        $('realLoadedCase'),
    modalityRow:       $('modalityRow'),
    orientationSelect: $('orientationSelect'),
    stages:            $('viewerStages'),
    sliceLabel:        $('viewerSliceLabel'),
    elevatorTrack:     $('elevatorTrack'),
    elevatorHandle:    $('elevatorHandle'),
    maskToggle:        $('maskToggle'),
    maskToggleState:   $('maskToggleState'),
    predToggle:        $('predToggle'),
    predToggleState:   $('predToggleState'),
    statVolume:        $('realStatVolume'),
    statSliceArea:     $('realStatSliceArea'),
    compositionBar:    $('realCompositionBar'),
    compositionLegend: $('realCompositionLegend'),
    compositionStat:   $('compositionStat'),
    predictionStats:   $('predictionStats'),
    predStatDice:      $('predStatDice'),
    predStatHd95:      $('predStatHd95'),
    predStatVolume:    $('predStatVolume')
  };

  let manifest = null;
  let modalityOrder = [];
  // One <canvas> per modality, built once (the channel set is fixed by the
  // manifest) and shown/hidden via the `hidden` attribute as the toggle row
  // is clicked — never torn down, so a re-enabled channel doesn't refetch.
  const canvasByModality = new Map(); // key -> { stage, canvas, ctx }
  // Cache key includes orientation: each viewing plane is a distinct atlas
  // PNG (axial/coronal/sagittal never share pixels), so a case/channel pair
  // has up to 3 independent cache entries, one fetched lazily per plane
  // actually visited.
  const imgCache = new Map(); // `${caseId}:${orientation}:${channelKeyOrTruth}` -> Promise<HTMLImageElement>

  //: Not radiologically verified (see config.ORIENTATION_* on the export
  //: side) — just what the slice-index readout is labelled with.
  const AXIS_LABEL = { axial: 'Z', coronal: 'Y', sagittal: 'X' };
  const ORIENTATION_KEYS = Object.keys(AXIS_LABEL);

  const state = {
    //: Which cohort the grid is showing. 'train' is the 60 labelled development
    //: cases; 'test' is the 60 unlabelled eval cases, which have a model
    //: prediction but no expert mask at all.
    split: 'train',
    caseId: null,
    orientation: 'axial',
    activeModalities: new Set(), // which channel panels are toggled on — all, by default
    sliceIndex: 0,
    showMask: true,
    showPrediction: false
  };

  const caseById = (id) => manifest.cases.find((c) => c.id === id) || null;
  //: Entries exported before the split existed carry no tag; they are the
  //: labelled development cohort, so treat a missing tag as 'train'.
  const splitOf = (c) => c.split || 'train';
  const casesInSplit = (split) => manifest.cases.filter((c) => splitOf(c) === split);
  const hasTruth = (c) => Boolean(c.masks[state.orientation] && c.masks[state.orientation].truth);
  const atlasInfo = () => manifest.atlas[state.orientation];

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`failed to load ${src}`));
      img.src = src;
    });
  }

  function ensureImage(caseId, orientation, key, src) {
    const cacheKey = `${caseId}:${orientation}:${key}`;
    if (!imgCache.has(cacheKey)) imgCache.set(cacheKey, loadImage(src));
    return imgCache.get(cacheKey);
  }

  /* ── drawing ─────────────────────────────────────────────────────────── */

  function tileRect(index) {
    const a = atlasInfo();
    const row = Math.floor(index / a.cols);
    const col = index % a.cols;
    return { sx: col * a.tile, sy: row * a.tile, s: a.tile };
  }

  //: Coronal/sagittal atlases come out of the exporter upside-down relative
  //: to how they read naturally in the viewer (see config.ORIENTATION_* on the
  //: export side — that axis convention was never verified against scanner
  //: orientation metadata). Axial is unaffected. Rotating at blit time, not in
  //: the exported PNGs, keeps this a pure display fix with zero risk of
  //: touching the atlas/manifest format the export pipeline owns.
  const ROTATE_180 = { coronal: true, sagittal: true };

  function blit(ctx, img, sx, sy, s, size, orientation) {
    if (!ROTATE_180[orientation]) {
      ctx.drawImage(img, sx, sy, s, s, 0, 0, size, size);
      return;
    }
    ctx.save();
    ctx.translate(size, size);
    ctx.rotate(Math.PI);
    ctx.drawImage(img, sx, sy, s, s, 0, 0, size, size);
    ctx.restore();
  }

  function draw() {
    const c = caseById(state.caseId);
    if (!c) return;

    const { sx, sy, s } = tileRect(state.sliceIndex);

    modalityOrder.forEach((key) => {
      const entry = canvasByModality.get(key);
      const active = state.activeModalities.has(key);
      entry.stage.hidden = !active;
      if (!active) return;

      const size = entry.canvas.width;
      // Snapshot what this draw is *for*. If the user toggles the channel off,
      // switches case/orientation, or scrubs on before the image resolves,
      // the stale draw bails instead of painting over a newer selection — the
      // only thing that makes rapid scrubbing (and rapid toggling) safe.
      const forCase = c.id, forOrientation = state.orientation, forModality = key, forSlice = state.sliceIndex;
      const stale = () =>
        state.caseId !== forCase || state.orientation !== forOrientation ||
        !state.activeModalities.has(forModality) || state.sliceIndex !== forSlice;

      ensureImage(forCase, forOrientation, forModality, c.channels[forOrientation][forModality]).then((img) => {
        if (stale()) return;
        entry.ctx.clearRect(0, 0, size, size);
        blit(entry.ctx, img, sx, sy, s, size, forOrientation);
        const truth = c.masks[forOrientation].truth;
        if (state.showMask && truth) {
          ensureImage(forCase, forOrientation, 'truth', truth).then((maskImg) => {
            if (stale() || !state.showMask) return;
            blit(entry.ctx, maskImg, sx, sy, s, size, forOrientation);
          });
        }
        // Ground truth (pink) and the model's prediction (cyan) are drawn as
        // two independent, differently-coloured overlays rather than a
        // separate pre-baked diff atlas — where they agree the colours mix,
        // where they don't, only one shows. That *is* the diff view, for free.
        const prediction = c.masks[forOrientation].prediction;
        if (state.showPrediction && prediction) {
          ensureImage(forCase, forOrientation, 'prediction', prediction).then((predImg) => {
            if (stale() || !state.showPrediction) return;
            blit(entry.ctx, predImg, sx, sy, s, size, forOrientation);
          });
        }
      });
    });

    updateSliceLabel(c);
  }

  function updateSliceLabel(c) {
    const a = atlasInfo();
    const z = a.z0 + state.sliceIndex * a.step;
    el.sliceLabel.textContent = `${AXIS_LABEL[state.orientation]} ${String(z).padStart(3, '0')}`;
    el.elevatorTrack.setAttribute('aria-valuenow', String(state.sliceIndex));
    el.elevatorTrack.setAttribute('aria-valuetext', `slice ${state.sliceIndex + 1} of ${a.n}`);
    el.elevatorHandle.style.top = `${a.n > 1 ? (state.sliceIndex / (a.n - 1)) * 100 : 0}%`;

    const areas = c.maskAreaBySlice ? c.maskAreaBySlice[state.orientation] : null;
    const area = areas ? areas[state.sliceIndex] : null;
    el.statSliceArea.textContent = (area === null || area === undefined)
      ? '—' : `${area.toLocaleString('en-US')} px²`;
  }

  /* ── stats / composition ─────────────────────────────────────────────── */

  const SUBLABEL_COLORS = { oedema: '#2a78d6', nonEnhancingCore: '#eb6834', enhancingTumour: '#1baf7a' };
  const SUBLABEL_NAMES  = { oedema: 'oedema', nonEnhancingCore: 'non-enhancing core', enhancingTumour: 'enhancing' };

  function renderStats(c) {
    // Test-set cases have no expert mask, so every truth-derived field is null.
    // Each is reported as unavailable rather than coerced: `null / 1000` would
    // render a confident "0.0 mL", which reads as "no tumour here" — the exact
    // opposite of "we have nothing to compare against".
    const sub = c.metrics.sublabelsMm3;
    const hasTruthMetrics = c.metrics.trueVolumeMm3 !== null &&
                            c.metrics.trueVolumeMm3 !== undefined && Boolean(sub);

    el.statVolume.textContent = hasTruthMetrics
      ? `${(c.metrics.trueVolumeMm3 / 1000).toFixed(1)} mL`
      : 'no expert mask';

    if (el.compositionStat) el.compositionStat.hidden = !hasTruthMetrics;
    if (!hasTruthMetrics) {
      el.compositionBar.replaceChildren();
      el.compositionLegend.replaceChildren();
      renderPredictionStats(c);
      return;
    }

    const total = sub.oedema + sub.nonEnhancingCore + sub.enhancingTumour || 1;
    const parts = Object.keys(SUBLABEL_NAMES).map((key) => ({ key, value: sub[key] }));

    el.compositionBar.replaceChildren(...parts.map(({ key, value }) => {
      const seg = document.createElement('span');
      seg.className = 'composition-bar__seg';
      seg.style.width = `${(100 * value / total).toFixed(1)}%`;
      seg.style.background = SUBLABEL_COLORS[key];
      return seg;
    }));

    el.compositionLegend.replaceChildren(...parts.map(({ key, value }) => {
      const li = document.createElement('li');
      li.className = 'composition-legend__item';
      const sw = document.createElement('span');
      sw.className = 'composition-legend__swatch';
      sw.style.background = SUBLABEL_COLORS[key];
      li.append(sw, document.createTextNode(`${SUBLABEL_NAMES[key]}: ${(value / 1000).toFixed(1)} mL`));
      return li;
    }));

    renderPredictionStats(c);
  }

  //: Whether the prediction OVERLAY exists and whether it can be SCORED are two
  //: different questions, and conflating them was a bug: gating the overlay
  //: toggle on `dice` hid the prediction on exactly the test cases where it is
  //: the only thing to look at. The toggle follows the mask; the accuracy
  //: numbers follow the score.
  function renderPredictionStats(c) {
    const anyPredMask = ORIENTATION_KEYS.some(
      (o) => c.masks[o] && c.masks[o].prediction);
    const hasScore = c.metrics.dice !== null && c.metrics.dice !== undefined;

    el.predToggle.hidden = !anyPredMask;
    el.predictionStats.hidden = !anyPredMask;
    if (!anyPredMask) return;

    el.predStatDice.textContent = hasScore ? c.metrics.dice.toFixed(3) : 'n/a';
    el.predStatHd95.textContent = (!hasScore || c.metrics.hd95 === null || Number.isNaN(c.metrics.hd95))
      ? 'n/a' : `${c.metrics.hd95.toFixed(1)} mm`;
    el.predStatVolume.textContent = (c.metrics.predictedVolumeMm3 === null ||
                                     c.metrics.predictedVolumeMm3 === undefined)
      ? '—' : `${(c.metrics.predictedVolumeMm3 / 1000).toFixed(1)} mL`;
    // The heading claims an out-of-fold comparison; on the test set there is
    // nothing to compare to, so say what is actually on screen.
    const heading = el.predictionStats.querySelector('.stat__label');
    if (heading) {
      heading.textContent = hasScore ? 'MODEL vs EXPERT (OUT-OF-FOLD)'
                                     : 'MODEL PREDICTION (NO EXPERT MASK)';
    }
  }

  /* ── modality switcher — independent toggles, all four on by default ──── */

  function buildStages() {
    el.stages.replaceChildren(...modalityOrder.map((key) => {
      const stage = document.createElement('div');
      stage.className = 'viewer__stage';
      stage.dataset.modality = key;

      const canvas = document.createElement('canvas');
      canvas.className = 'viewer__canvas';
      canvas.width = 384;
      canvas.height = 384;
      canvas.tabIndex = 0;
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `${manifest.modalities[key]} slice viewer`);

      const tag = document.createElement('span');
      tag.className = 'viewer__tag';
      tag.textContent = manifest.modalities[key];

      stage.append(canvas, tag);
      canvasByModality.set(key, { stage, canvas, ctx: canvas.getContext('2d') });
      return stage;
    }));
  }

  function toggleModality(key) {
    if (state.activeModalities.has(key)) state.activeModalities.delete(key);
    else state.activeModalities.add(key);
    syncModalityButtons();
    draw();
  }

  function renderModalityRow() {
    el.modalityRow.replaceChildren(...modalityOrder.map((key) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'modality-btn';
      btn.dataset.modality = key;
      btn.setAttribute('role', 'checkbox');
      btn.setAttribute('aria-checked', String(state.activeModalities.has(key)));
      btn.setAttribute('aria-label', `${manifest.modalities[key]} channel`);
      btn.textContent = manifest.modalities[key];
      btn.addEventListener('click', () => toggleModality(key));
      return btn;
    }));
  }

  function syncModalityButtons() {
    el.modalityRow.querySelectorAll('.modality-btn').forEach((btn) => {
      btn.setAttribute('aria-checked', String(state.activeModalities.has(btn.dataset.modality)));
    });
  }

  /* ── case grid ───────────────────────────────────────────────────────── */

  function renderCaseGrid() {
    el.caseGrid.replaceChildren(...casesInSplit(state.split).map((c) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'case-card';
      card.dataset.caseId = c.id;
      card.setAttribute('role', 'radio');
      card.setAttribute('aria-checked', 'false');

      const img = document.createElement('img');
      img.className = 'case-card__thumb';
      img.src = c.thumb;
      img.alt = '';
      img.loading = 'lazy';

      const name = document.createElement('span');
      name.className = 'case-card__name';
      name.textContent = c.label;

      card.append(img, name);
      card.addEventListener('click', () => selectCase(c.id));
      return card;
    }));
  }

  function syncCaseGridSelection() {
    el.caseGrid.querySelectorAll('.case-card').forEach((card) => {
      card.setAttribute('aria-checked', String(card.dataset.caseId === state.caseId));
    });
  }

  function selectCase(id) {
    const c = caseById(id);
    if (!c) return;

    state.caseId = id;
    // Channel toggles and the chosen viewing plane are viewing preferences,
    // not per-case ones — they carry over across case switches, unlike the
    // mask toggle below, which is deliberately reset to on for every newly
    // loaded case. bestIndex is keyed by orientation because "largest lesion
    // cross-section" means a different slice on each plane.
    state.sliceIndex = c.bestIndex[state.orientation];
    // Model prediction stays locked behind the paywall per case, UNLESS the
    // user already "paid" this session — that unlock is global (a fake
    // subscription, not a per-case purchase), so it carries over here too.
    state.showPrediction = !!(NS.paywall && NS.paywall.isPaid());

    syncCaseGridSelection();
    syncModalityButtons();
    el.predToggle.setAttribute('aria-pressed', String(state.showPrediction));
    el.predToggleState.textContent = state.showPrediction ? 'ON' : 'OFF';

    // A test case has no expert mask, so the EXPERT MASK control has nothing
    // to draw. Disable and relabel it rather than leaving a button that looks
    // live and silently does nothing.
    const truth = hasTruth(c);
    state.showMask = truth;
    el.maskToggle.disabled = !truth;
    el.maskToggle.setAttribute('aria-pressed', String(truth));
    el.maskToggleState.textContent = truth ? 'ON' : 'NONE';
    el.maskToggle.title = truth ? '' : 'No expert mask exists for the test set';

    el.emptyState.hidden = true;
    el.panel.hidden = false;
    el.loadedCase.textContent = `${c.label} · 4 channels · 155 slices`;
    el.elevatorTrack.setAttribute('aria-valuemax', String(atlasInfo().n - 1));

    renderStats(c);
    draw();
  }

  /* ── train / test switch ─────────────────────────────────────────────── */

  function setSplit(split) {
    if (split === state.split || !casesInSplit(split).length) return;
    state.split = split;
    state.caseId = null;

    // Collapse back to the empty state: the previously loaded case belongs to
    // the other cohort, and silently keeping it on screen under a new heading
    // would misrepresent which set is being viewed.
    el.panel.hidden = true;
    el.emptyState.hidden = false;

    renderCaseGrid();
    syncSplitSwitch();
  }

  function syncSplitSwitch() {
    if (!el.splitSwitch) return;
    el.splitSwitch.querySelectorAll('.split-switch__btn').forEach((btn) => {
      const s = btn.dataset.split;
      const n = casesInSplit(s).length;
      btn.setAttribute('aria-checked', String(s === state.split));
      btn.disabled = n === 0;
      if (n === 0) btn.title = `No ${s} cases have been exported`;
    });
    if (el.scannerBlurb) {
      const n = casesInSplit(state.split).length;
      el.scannerBlurb.textContent = state.split === 'train'
        ? `${n} labelled cases · 4 MRI channels each · scroll to move through the brain.`
        : `${n} held-out test cases · no expert mask exists · model prediction only.`;
    }
    if (el.emptyState) {
      el.emptyState.textContent = state.split === 'train'
        ? 'No case loaded. Pick a patient file above.'
        : 'No case loaded. Pick a test scan above.';
    }
  }

  function bindSplitSwitch() {
    if (!el.splitSwitch) return;
    el.splitSwitch.addEventListener('click', (e) => {
      const btn = e.target.closest('.split-switch__btn');
      if (btn && !btn.disabled) setSplit(btn.dataset.split);
    });
  }

  /* ── orientation switcher ─────────────────────────────────────────────── */

  function setOrientation(orientation) {
    if (orientation === state.orientation || !manifest.atlas[orientation]) return;
    state.orientation = orientation;

    const c = caseById(state.caseId);
    // Re-anchor on the slice with the largest lesion cross-section *for this
    // plane* — carrying over the raw index would land on an arbitrary slice,
    // since axial/coronal/sagittal don't share an index space.
    state.sliceIndex = c ? c.bestIndex[orientation] : 0;

    el.elevatorTrack.setAttribute('aria-label', `Slice, along the ${orientation} plane`);
    el.elevatorTrack.setAttribute('aria-valuemax', String(atlasInfo().n - 1));
    if (c) draw();
  }

  function bindOrientationSelect() {
    el.orientationSelect.value = state.orientation;
    el.orientationSelect.addEventListener('change', () => setOrientation(el.orientationSelect.value));
  }

  /* ── slice navigation ────────────────────────────────────────────────── */

  function setSlice(index) {
    if (!state.caseId) return;
    const n = atlasInfo().n;
    state.sliceIndex = Math.max(0, Math.min(n - 1, Math.round(index)));
    draw();
  }

  function bindSliceControls() {
    // Wheel/keydown are bound once on the stages *container*, not per canvas —
    // there can be up to 4 of them and any one may be toggled away mid-session,
    // so delegation (both events bubble) is what makes "scroll over whichever
    // channel is focused" keep working as panels come and go.
    //
    // Vertical gesture, so it does not collide with the legacy horizontal
    // compare-slider drag (which lives on a different element entirely on
    // this page).
    el.stages.addEventListener('wheel', (e) => {
      if (!state.caseId) return;
      e.preventDefault();
      setSlice(state.sliceIndex + (e.deltaY > 0 ? 1 : -1));
    }, { passive: false });

    const pctFromClientY = (clientY) => {
      const rect = el.elevatorTrack.getBoundingClientRect();
      return (clientY - rect.top) / rect.height;
    };
    const sliceFromPct = (pct) => pct * (atlasInfo().n - 1);

    el.elevatorTrack.addEventListener('pointerdown', (e) => {
      if (!state.caseId) return;
      el.elevatorTrack.setPointerCapture(e.pointerId);
      setSlice(sliceFromPct(pctFromClientY(e.clientY)));
      e.preventDefault();
    });
    el.elevatorTrack.addEventListener('pointermove', (e) => {
      if (!el.elevatorTrack.hasPointerCapture(e.pointerId)) return;
      setSlice(sliceFromPct(pctFromClientY(e.clientY)));
    });

    function keyStep(e) {
      if (!state.caseId) return;
      if (e.key === 'Home') { e.preventDefault(); setSlice(0); return; }
      if (e.key === 'End') { e.preventDefault(); setSlice(atlasInfo().n - 1); return; }
      const step = e.shiftKey ? 5 : 1;
      const moves = { ArrowUp: -step, ArrowDown: step, PageUp: -10, PageDown: 10 };
      if (!(e.key in moves)) return;
      e.preventDefault();
      setSlice(state.sliceIndex + moves[e.key]);
    }
    el.elevatorTrack.addEventListener('keydown', keyStep);
    el.stages.addEventListener('keydown', keyStep); // a reader who tabs to any channel canvas instead
  }

  function bindMaskToggles() {
    el.maskToggle.addEventListener('click', () => {
      state.showMask = !state.showMask;
      el.maskToggle.setAttribute('aria-pressed', String(state.showMask));
      el.maskToggleState.textContent = state.showMask ? 'ON' : 'OFF';
      draw();
    });
    function setPrediction(on) {
      state.showPrediction = on;
      el.predToggle.setAttribute('aria-pressed', String(on));
      el.predToggleState.textContent = on ? 'ON' : 'OFF';
      draw();
    }

    el.predToggle.addEventListener('click', () => {
      // Turning it off is always free. Turning it on requires the fake
      // subscription — already paid this session just flips it straight on.
      if (state.showPrediction) { setPrediction(false); return; }
      if (NS.paywall) NS.paywall.open(() => setPrediction(true));
      else setPrediction(true); // no paywall markup on this page — fail open
    });
  }

  /* ── boot ────────────────────────────────────────────────────────────── */

  function initReal(loadedManifest) {
    manifest = loadedManifest;
    modalityOrder = Object.keys(manifest.modalities);
    modalityOrder.forEach((key) => state.activeModalities.add(key)); // all on by default

    if (legacyLab) legacyLab.hidden = true;
    realLab.hidden = false;

    buildStages();
    renderModalityRow();
    renderCaseGrid();
    syncSplitSwitch();
    bindSplitSwitch();
    bindSliceControls();
    bindMaskToggles();
    bindOrientationSelect();
  }

  fetch('assets/cases/manifest.json', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error('manifest not found'))))
    .then((m) => {
      if (!m || !Array.isArray(m.cases) || !m.cases.length) throw new Error('manifest has no cases');
      initReal(m);
    })
    .catch(() => {
      // No dataset export (yet) — leave the legacy mock demo app.js already
      // rendered exactly as it is. Documented fallback, not an error.
    });

  NS.viewer = { selectCase, setSlice, setSplit };

})(window.TumorNet);
