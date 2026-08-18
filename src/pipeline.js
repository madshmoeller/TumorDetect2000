/* ═══════════════════════════════════════════════════════════════════════
   The detection seam.

   Everything the UI knows about "detecting a tumor" goes through this one
   contract. Swap the implementation and the GUI does not change.

   ── contract ──────────────────────────────────────────────────────────

   pipeline.detect(scan, options) -> Promise<Detection>

     scan                one entry from TumorNet.scans
     options.onProgress  fn({ pct, message }) — called zero or more times.
                         pct 0..100. message may be null.
     options.signal      optional AbortSignal. Reject with an AbortError
                         when it fires; the UI treats that as "cancelled",
                         not "failed".

   Detection = {
     confidence  number    0..1
     volumeMm3   number    estimated tumour volume, cubic millimetres
     verdict     string    short human-readable call
     overlays    {         image URLs, each drawn on top of scan.base
       truth        string|null    reference / ground-truth mask
       prediction   string        the model's mask
       difference   string|null   agreement map (see legend in index.html)
     }
     quip        string|null   mascot line; null -> UI stays quiet
   }

   Rejecting the promise is a legitimate outcome — the UI shows the failure
   and resets the button. Do not resolve with a half-filled Detection.
   ═══════════════════════════════════════════════════════════════════════ */

window.TumorNet = window.TumorNet || {};

(function (NS) {
  'use strict';

  function abortError() {
    // DOMException is the shape callers expect from a real fetch abort.
    return new DOMException('Detection aborted', 'AbortError');
  }

  /**
   * Stand-in detector. Replays the fixture's known answer behind a fake
   * progress ticker, so the interaction is real even though the maths is not.
   *
   * Timing matches the prototype: a tick every 200ms adding 7–18%, so a run
   * lands somewhere around 1.5–3 seconds.
   */
  NS.createMockPipeline = function createMockPipeline(options) {
    const opts = options || {};
    const tickMs = opts.tickMs || 200;
    const messages = opts.messages || NS.progressMessages;

    return {
      name: 'mock',

      detect(scan, runOptions) {
        const { onProgress, signal } = runOptions || {};

        return new Promise((resolve, reject) => {
          if (signal && signal.aborted) return reject(abortError());

          let pct = 0;
          let timer = null;

          const stop = () => {
            if (timer !== null) clearInterval(timer);
            timer = null;
            if (signal) signal.removeEventListener('abort', onAbort);
          };

          const onAbort = () => { stop(); reject(abortError()); };
          if (signal) signal.addEventListener('abort', onAbort, { once: true });

          const messageFor = (p) => messages[
            Math.min(messages.length - 1, Math.floor((p / 100) * messages.length))
          ];

          if (onProgress) onProgress({ pct: 0, message: messages[0] });

          timer = setInterval(() => {
            pct = Math.min(100, pct + 7 + Math.random() * 11);
            if (onProgress) {
              onProgress({ pct, message: pct < 100 ? messageFor(pct) : null });
            }
            if (pct < 100) return;

            stop();
            // Brief beat at 100% before results appear, as in the prototype.
            setTimeout(() => {
              if (signal && signal.aborted) return;
              const e = scan.expected;
              if (!e) {
                reject(new Error(`No fixture result for scan "${scan.id}"`));
                return;
              }
              resolve({
                confidence: e.confidence,
                volumeMm3: e.volumeMm3,
                verdict: e.verdict,
                overlays: {
                  truth: e.overlays.truth || null,
                  prediction: e.overlays.prediction,
                  difference: e.overlays.difference || null
                },
                quip: e.quip || null
              });
            }, 200);
          }, tickMs);
        });
      }
    };
  };

  /* The active pipeline. Reassign this to go live, e.g.
       TumorNet.pipeline = TumorNet.createHttpPipeline('/api/detect');
     as long as the replacement satisfies the contract above. */
  NS.pipeline = NS.createMockPipeline();

})(window.TumorNet);
