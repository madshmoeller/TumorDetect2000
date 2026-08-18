/* ═══════════════════════════════════════════════════════════════════════
   Demo fixtures — patient files and mascot copy.

   These are the three canned scans from the design prototype. When the real
   pipeline lands, `scans` becomes whatever the backend lists (see
   pipeline.js) and `truth` / `expected` drop out — they only exist so the
   mock detector has something to "predict".
   ═══════════════════════════════════════════════════════════════════════ */

window.TumorNet = window.TumorNet || {};

(function (NS) {
  'use strict';

  NS.scans = [
    {
      id: 'scan1',
      filename: 'brainscan_001_FINAL.jpg',
      base: 'assets/scan1_base.png',
      // What the mock detector will "find". A real pipeline returns these.
      expected: {
        confidence: 0.76,
        volumeMm3: 3160,
        verdict: 'Cautiously Okay',
        overlays: {
          truth: 'assets/scan1_gt.png',
          prediction: 'assets/scan1_pred.png',
          difference: 'assets/scan1_diff.png'
        },
        quip: "76% confident. That's almost reasonable — I'm as surprised as you are. " +
              "Still wouldn't bet your life on it. But don't listen to me, I'm a Cancer."
      }
    },
    {
      id: 'scan2',
      filename: 'brainscan_002_final_v2.jpg',
      base: 'assets/scan2_base.png',
      expected: {
        confidence: 0.91,
        volumeMm3: 3730,
        verdict: 'Confidently Meh',
        overlays: {
          truth: 'assets/scan2_gt.png',
          prediction: 'assets/scan2_pred.png',
          difference: 'assets/scan2_diff.png'
        },
        quip: '91% confident, and only sort of correct. Peak machine learning energy right there. ' +
              "Don't listen to me though — I'm a Cancer."
      }
    },
    {
      id: 'scan3',
      filename: 'brainscan_003_REALFINAL.jpg',
      base: 'assets/scan3_base.png',
      expected: {
        confidence: 0.999,
        volumeMm3: 3390,
        verdict: 'Beautifully Wrong',
        overlays: {
          truth: 'assets/scan3_gt.png',
          prediction: 'assets/scan3_pred.png',
          difference: 'assets/scan3_diff.png'
        },
        quip: '99.9% confident and completely off. I respect the confidence, I question everything else. ' +
              "Then again, don't listen to me — I'm a Cancer, zodiacally speaking."
      }
    }
  ];

  /* Shown in sequence while a detection runs. The pipeline picks which one
     is current from the progress percentage. */
  NS.progressMessages = [
    'Warming up the neurons...',
    "Consulting my gut. I don't have one.",
    'Running very serious math...',
    'Asking the Magic 8-Ball for a second opinion...',
    'Finalizing a confident-sounding number...'
  ];

  /* Cycled through on every click of the mascot. */
  NS.trivia = [
    (name) => `Fun fact: ${name} is not a licensed medical device. ${name} is barely a licensed brain.`,
    ()     => "I don't do IoU or Dice scores anymore. Too many carbs.",
    ()     => 'Horoscope for today: mild confidence, moderate dread. Standard Tuesday for a Cancer.',
    ()     => 'This whole site runs on vibes and one (1) neural net.',
    (name) => `${name} contains multitudes. Mostly regret and gray matter.`,
    ()     => 'Please consult an actual radiologist. I am a decorative brain.'
  ];

  NS.mascotLines = {
    greeting: (name) =>
      `Hey. I'm ${name}. I live in this corner now. Fair warning: I'm a Cancer, so don't trust a word I say.`,
    scanLoaded: () =>
      "Ooh, fresh pixels loaded. Hit DETECT TUMOR when you're ready, champ.",
    noScan: () =>
      "Pick a scan first. I'm a brain, not a mind reader — and don't trust my mind either, I'm a Cancer.",
    guestbook: () =>
      "This guestbook is READ-ONLY. Much like my opinions, which you should not act on. I'm a Cancer.",
    failed: () =>
      "Something broke and it probably wasn't my fault. Try again? I'm a Cancer, I don't handle blame well.",
    paywall: () =>
      "Great news, your tumor is done! Bad news, it's premium content now. I don't make the rules, I'm a Cancer."
  };

  NS.guestbookEntries = [
    { name: 'xXradiologist99Xx', text: "this site has better graphics than my hospital’s actual PACS system!! 5 stars" },
    { name: 'tumor_stan_2000',   text: 'is it normal that the brain guy winked at me. asking for a friend' },
    { name: 'anonymous',         text: 'model said 99% confident and it was very wrong but the font was great so whatever' },
    { name: 'webmaster',         text: 'please stop feeding the brain mascot, he is getting ideas' },
    { name: 'DialUpDan',         text: 'downloaded this whole site over 56k just to hear the brain call itself a Cancer again' },
    { name: 'skeptical_intern',  text: 'showed my attending the confidence scores and he just stared at the wall for a while' },
    { name: 'MRI_enjoyer',       text: 'the diff overlay is oddly soothing. also terrifying. 10/10' },
    { name: 'notADoctor42',      text: 'i do not have a medical degree and this site has still taught me nothing, perfect' },
    { name: 'geocities_refugee', text: 'finally a website that respects my bandwidth and my low expectations' },
    { name: 'concerned_mom',     text: "made my son promise he'd see a real doctor and not just trust the cartoon brain" },
    { name: 'agentic_andy',      text: "this is NOT vibe coded, it is agentically engineered, and if you disagree i will send an agent to your house" },
    { name: 'vibe_coder_vic',    text: "i vibe coded a competing tumor detector in an afternoon. it just says 'probably fine' every time. taking meetings" },
    { name: 'HIPAA_but_ok',      text: "uploaded my actual scan by accident, brain guy said '99% confident, nice knees' so i think we're fine" },
    { name: 'popup_survivor',    text: "the ads spawn faster than i can close them and somehow i respect that more than the model's accuracy" },
    { name: 'residency_year1',   text: "showed this to the whole cohort during rounds, program director did not laugh, worth it" },
    { name: 'Cancer_the_sign',   text: "as a fellow Cancer i relate to braino's commitment to not trusting himself" },
    { name: 'lurker_since_2003', text: "been refreshing this page for hours, hit counter went up by exactly 1, that was me twice" }
  ];

})(window.TumorNet);
