# How the tumour-finding model works, in plain terms

No prior knowledge of MRI or image segmentation assumed.

## The task

Each patient has four MRI scans of the same brain, taken with different settings
(called "channels" or "modalities" — think of them as four camera filters over
the same scene: one makes fluid bright, another makes anatomy clear, one uses a
contrast dye). Doctors have already drawn, by hand, exactly which part of the
brain is tumour in each of 60 patients.

The job: teach a computer to draw that same outline on its own, on a patient it
has never seen. That is **segmentation** — labelling every point in the image as
"tumour" or "not tumour", rather than just saying "there is a tumour somewhere".

An MRI scan is not one flat photo. It is a stack of thin cross-sections through
the head, like slices of bread — 155 of them here.

```
          ONE PATIENT = FOUR STACKED 3D VOLUMES

     FLAIR          T1w           T1-Gd           T2w
   ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
   │▒▒░░▒▒▒│     │███▓▓██│     │███▓▓██│     │░░▒▒░░░│   240 × 240 pixels
   │▒░███░▒│ ╲   │██░░░██│ ╲   │██▓█▓██│   ╱ │░▒███▒░│   × 155 slices
   │▒░███░▒│  ╲  │██░░░██│  ╲  │██▓█▓██│  ╱  │░▒███▒░│
   │▒▒░░▒▒▒│   ╲ │███▓▓██│   ╲ │███▓▓██│ ╱   │░░▒▒░░░│   the SAME brain,
   └───────┘    ╲└───────┘    ╲└───────┘╱    └───────┘   four different
                 ╲             ╲       ╱                 "filters"
                  ╲             ╲     ╱
                   ▼             ▼   ▼
              ┌──────────────────────────┐
              │   the model sees all 4   │  →  one tumour outline
              │   at once, in 3D         │
              └──────────────────────────┘
```

## The model: a "U-Net"

The model is a **U-Net**, named for the shape of its diagram: it narrows down,
then widens back out, like the letter U. Ours works on 3D blocks of brain
(96 × 160 × 160 voxels at a time), not flat slices.

```
        ENCODER — shrink                                    DECODER — grow back
   "give up detail, gain context"                     "restore detail, keep context"

 4ch ─┬─[  20  ]── 96×160×160 ────────── skip ──────────▶ [  20  ]─┬─▶ ◉ MAIN OUTPUT
      │     │                                                ▲     │    the answer
      │  shrink ½                                          grow ×2  │
      │     ▼                                                │     │
      │  [  40  ]── 48× 80× 80 ────────── skip ──────────▶ [  40  ]─┼─▶ ◎ helper (½ size)
      │     │                                                ▲     │
      │  shrink ½                                          grow ×2  │
      │     ▼                                                │     │
      │  [  80  ]── 24× 40× 40 ────────── skip ──────────▶ [  80  ]─┴─▶ ◎ helper (¼ size)
      │     │                                                ▲
      │  shrink ½                                          grow ×2
      │     ▼                                                │
      │  [ 160  ]── 12× 20× 20 ────────── skip ──────────▶ [ 160  ]
      │     │                                                ▲
      │  shrink ½                                          grow ×2
      └──▶[ 320  ]──  6× 10× 10  ═══ THE BOTTLENECK ═══════════┘
                    "sees the whole neighbourhood at once,
                     but far too blurry to draw an outline"

      [ N ] = how many different pattern-detectors ("filters") live at that level
```

Read it as a journey. Going **down** the left side, the image gets smaller and
blurrier but each step sees *more of the surrounding brain* — useful for "is this
near the middle? near the skull?". At the bottom the model understands the whole
region but has lost the fine edges. Going **up** the right side it rebuilds the
detail.

### Why the horizontal arrows matter

Those `skip` arrows are the trick that makes a U-Net work. Without them, all the
fine detail thrown away on the way down is gone forever:

```
   WITHOUT skips                         WITH skips
   ─────────────                         ──────────
   sharp ──▶ blurry ──▶ ??? sharp        sharp ──┬──▶ blurry ──▶ sharp
                                                 └── "here is where the
   the model must invent the edges                    edges actually were" ──┘
   back from a blurry summary            the model is handed them back
```

### The two "helper" outputs (deep supervision)

Notice the `◎ helper` arrows. Besides the real answer, the model is also asked to
produce a rough, half-size and quarter-size sketch of the tumour, and is scored
on those too.

```
   full size  96×160×160   ◉ ████████   the answer we keep   — 57% of the score
   half size  48× 80× 80   ◎ ▓▓▓▓       rough sketch         — 29%
   quarter    24× 40× 40   ◎ ▒▒         very rough sketch    — 14%
```

Why bother? Because the correction signal that teaches the network has to travel
all the way back from the output to the deepest layers, and it gets weak on long
journeys. The helpers inject fresh guidance directly into the middle of the
network — like giving feedback at each stage of a drawing instead of only at the
end. **The helpers are used only during training and thrown away afterwards.**

## How it is trained and checked, honestly

Only 60 patients exist. To get a fair read on all 60 without ever testing on a
patient the model trained on, the data is split into 5 groups:

```
   60 patients ──▶ ┌────┬────┬────┬────┬────┐   12 each
                   │ A  │ B  │ C  │ D  │ E  │
                   └────┴────┴────┴────┴────┘

   run 1:  train B C D E  ──▶ test A      ┐
   run 2:  train A C D E  ──▶ test B      │  every patient is tested
   run 3:  train A B D E  ──▶ test C      │  exactly once, always by a
   run 4:  train A B C E  ──▶ test D      │  model that never saw them
   run 5:  train A B C D  ──▶ test E      ┘
```

- **A target score was written down before training started, not after seeing the
  result.** The score is *Dice*: 1.0 means the outline matches the doctor's
  exactly, 0.0 means no overlap. A number is only trustworthy if someone decided
  in advance what counts as good.
- Two much simpler methods are scored on the same splits for comparison — one that
  just looks for bright spots on a single channel, one classical statistical
  method. If the U-Net could not beat those, that would be worth knowing, not
  hiding.
- With only 60 patients the model is deliberately handicapped during training —
  random flips, small rotations, brightness changes, simulated scanner
  unevenness — so it learns what a tumour looks like instead of memorising 48
  brains.

## For a brand-new patient: all five models vote

The five runs above leave us with five trained models. For a *new* scan that none
of them has seen, we can use all five and average their answers, which is more
reliable than any one alone:

```
   new scan ──┬──▶ model A ──▶ ▒▒▒▒  ┐
              ├──▶ model B ──▶ ▒▒▒▒  │
              ├──▶ model C ──▶ ▒▒▒▒  ├──▶ average ──▶ ████ final outline
              ├──▶ model D ──▶ ▒▒▒▒  │
              └──▶ model E ──▶ ▒▒▒▒  ┘
```

This is **not** used for the 60 development patients, and that restraint matters:
four of the five models trained on any given one of them, so averaging would be
marking your own homework. It is only valid for genuinely unseen scans.

## Where it stands

| | Dice |
|---|---|
| bright-spot baseline | 0.680 |
| classical statistical baseline | 0.711 |
| **this model, on patients it never saw** | **0.905** |

Median 0.928. The weakest single case is 0.612 — small tumours are the hard ones,
because getting a small outline slightly wrong costs proportionally much more.

Every number here is out-of-fold, and the pre-registered figure fixed before any
result was seen was 0.887; see `RULES.md` for the targets and
`outputs/prereg_20260817/README.md` for what changed afterwards and why.

See `docs/architecture.md` for the technical version, and `docs/model_scope.md`
for the compute decisions.
