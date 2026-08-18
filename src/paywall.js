/* ═══════════════════════════════════════════════════════════════════════
   The fake payment wall — shared by the real viewer (viewer.js) and the
   legacy mock demo (app.js). Neither of them owns payment state; both just
   call NS.paywall.isPaid() / NS.paywall.open(onSuccess).

   Entirely client-side and disposable: nothing typed into the form is read
   anywhere but the disabled-state check below, and nothing is ever sent,
   stored, or charged. "Paid" just means "clicked the button once" — it is
   not per-case, matching the fake $19.99/mo subscription copy: unlock once,
   model predictions stay visible everywhere for the rest of the session.
   ═══════════════════════════════════════════════════════════════════════ */

window.TumorNet = window.TumorNet || {};

(function (NS) {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const el = {
    panel:   $('paywallPanel'),
    card:    $('paywallCard'),
    expiry:  $('paywallExpiry'),
    cvc:     $('paywallCvc'),
    name:    $('paywallName'),
    payBtn:  $('paywallPayBtn'),
    cancelBtn: $('paywallCancelBtn')
  };
  if (!el.panel) return; // page has no paywall markup — nothing to wire up

  let paid = false;
  let onSuccess = null;

  const fields = () => [el.card, el.expiry, el.cvc, el.name];

  function updatePayButton() {
    el.payBtn.disabled = !fields().every((input) => input.value.trim().length > 0);
  }

  function resetForm() {
    fields().forEach((input) => { input.value = ''; });
    updatePayButton();
  }

  function close() {
    el.panel.hidden = true;
    onSuccess = null;
  }

  function open(onSuccessCb) {
    if (paid) { if (onSuccessCb) onSuccessCb(); return; }
    onSuccess = onSuccessCb || null;
    resetForm();
    el.panel.hidden = false;
  }

  function pay() {
    if (el.payBtn.disabled) return;
    paid = true;
    resetForm();
    el.panel.hidden = true;
    const cb = onSuccess;
    onSuccess = null;
    if (cb) cb();
    if (NS.say) NS.say('Payment "processed." TumorNet Payment Systems ™ does not exist, but your model prediction does.');
  }

  el.payBtn.addEventListener('click', pay);
  if (el.cancelBtn) el.cancelBtn.addEventListener('click', close);
  fields().forEach((input) => input.addEventListener('input', updatePayButton));
  updatePayButton();

  NS.paywall = {
    isPaid: () => paid,
    open
  };

})(window.TumorNet);
