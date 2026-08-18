"""Exploratory data analysis: figures F01-F05, and the numbers behind them.

Rule 2 says never accept a number you have not seen a figure for. This module is
where the numbers and the figures are produced by the same code path, so they
cannot drift apart.

    python -m ml.eda            # compute stats (cached) and draw F01-F05
    python -m ml.eda --verify   # re-derive the headline numbers and assert them
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config as C
from . import data as D
from . import figstyle as S

STATS_PATH = C.OUTPUTS / "eda_stats.json"

RAW_BINS = np.linspace(0, 2200, 161)
NRM_BINS = np.linspace(-3, 6, 161)


# ── statistics ─────────────────────────────────────────────────────────────


def compute_stats(case_ids: list[str], *, verbose: bool = True) -> dict:
    """One pass over the raw volumes, collecting everything the figures need.

    Histograms rather than raw voxels: 60 cases x 1.4M brain voxels x 4 channels
    will not fit in memory, but a 160-bin histogram per case per channel is 300 kB
    for the whole cohort and is all a density plot ever needed.
    """
    per_case = []
    spatial = np.zeros((C.CROP_D, *C.CROP_HW), dtype=np.float32)
    slice_area = np.zeros((len(case_ids), C.VOLUME_SHAPE[2]), dtype=np.int32)

    for i, cid in enumerate(case_ids):
        if verbose:
            print(f"\r  scanning {i + 1}/{len(case_ids)}  {cid}", end="", flush=True)

        images, labels = D.load_raw(cid)
        brain = D.brain_mask(images)
        normed = D.normalise(images, brain)
        tumour = labels > 0

        raw_hist, nrm_hist, brain_mean, brain_std = [], [], [], []
        for c in range(C.N_CHANNELS):
            vals = images[c][brain]
            raw_hist.append(np.histogram(vals, bins=RAW_BINS, density=True)[0].tolist())
            nrm_hist.append(np.histogram(normed[c][brain], bins=NRM_BINS, density=True)[0].tolist())
            brain_mean.append(float(vals.mean()))
            brain_std.append(float(vals.std()))

        counts = np.bincount(labels.ravel(), minlength=4)
        per_slice = tumour.sum(axis=(1, 2))
        slice_area[i] = per_slice
        spatial += D.crop(tumour).astype(np.float32)

        per_case.append(
            {
                "case": cid,
                "brain_voxels": int(brain.sum()),
                "tumour_voxels": int(tumour.sum()),
                "sublabels": {str(k): int(counts[k]) for k in (1, 2, 3)},
                "positive_slices": int((per_slice > 0).sum()),
                "raw_hist": raw_hist,
                "nrm_hist": nrm_hist,
                "brain_mean": brain_mean,
                "brain_std": brain_std,
            }
        )

    if verbose:
        print()

    spatial /= len(case_ids)
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.save(C.OUTPUTS / "spatial_prior.npy", spatial)
    np.save(C.OUTPUTS / "slice_area.npy", slice_area)

    return {"n_cases": len(case_ids), "per_case": per_case}


def load_stats(case_ids: list[str], *, force: bool = False) -> dict:
    if not force and STATS_PATH.exists():
        stats = json.loads(STATS_PATH.read_text())
        if stats.get("n_cases") == len(case_ids):
            return stats
    stats = compute_stats(case_ids)
    STATS_PATH.write_text(json.dumps(stats))
    return stats


def headline(stats: dict) -> dict:
    """The numbers this project quotes. Every one appears in a figure."""
    pc = stats["per_case"]
    tv = np.array([c["tumour_voxels"] for c in pc])
    bv = np.array([c["brain_voxels"] for c in pc])
    total = int(np.prod(C.VOLUME_SHAPE))
    slice_area = np.load(C.OUTPUTS / "slice_area.npy")
    pos = slice_area[slice_area > 0]

    return {
        "n_cases": len(pc),
        "tumour_voxels": {
            "min": int(tv.min()),
            "p25": int(np.percentile(tv, 25)),
            "median": int(np.median(tv)),
            "p75": int(np.percentile(tv, 75)),
            "max": int(tv.max()),
        },
        "burden_ratio_max_min": float(tv.max() / tv.min()),
        "brain_fraction_mean": float((bv / total).mean()),
        "tumour_frac_volume_median": float(np.median(tv / total)),
        "tumour_frac_brain_median": float(np.median(tv / bv)),
        "positive_slices": int((slice_area > 0).sum()),
        "total_slices": int(slice_area.size),
        "positive_slice_fraction": float((slice_area > 0).mean()),
        "positive_slice_area": {
            "p10": int(np.percentile(pos, 10)),
            "median": int(np.median(pos)),
            "p90": int(np.percentile(pos, 90)),
        },
        "tiny_positive_slice_fraction": float((pos < 100).mean()),
        "flair_brain_mean_range": [
            float(min(c["brain_mean"][0] for c in pc)),
            float(max(c["brain_mean"][0] for c in pc)),
        ],
        "flair_brain_mean_ratio": float(
            max(c["brain_mean"][0] for c in pc) / min(c["brain_mean"][0] for c in pc)
        ),
    }


# ── figures ────────────────────────────────────────────────────────────────


def fig01_cohort(stats: dict, hl: dict) -> None:
    import matplotlib.pyplot as plt

    pc = sorted(stats["per_case"], key=lambda c: c["tumour_voxels"])
    vol = np.array([c["tumour_voxels"] for c in pc]) / 1000.0  # mL, 1 mm iso
    sub = {k: np.array([c["sublabels"][str(k)] for c in pc]) / 1000.0 for k in (1, 2, 3)}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.42}
    )

    ax1.hist(vol, bins=18, color=S.SERIES[0], edgecolor=S.PAPER, linewidth=1.2)
    med = float(np.median(vol))
    ax1.axvline(med, color=S.INK, linewidth=1.5, linestyle="--")
    ax1.annotate(
        f"median {med:.0f} mL",
        xy=(med, ax1.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=S.INK,
        fontweight="bold",
    )
    ax1.set_title("a · Total tumour volume per case")
    ax1.set_xlabel("tumour volume (mL)")
    ax1.set_ylabel("cases")

    x = np.arange(len(pc))
    bottom = np.zeros(len(pc))
    for k in (1, 2, 3):
        ax2.bar(
            x, sub[k], bottom=bottom, width=0.82,
            color=S.SUBLABEL_COLORS[k], label=C.SUBLABELS[k],
            edgecolor=S.PAPER, linewidth=0.6,
        )
        bottom += sub[k]
    ax2.set_title("b · Composition, cases sorted by total burden")
    ax2.set_xlabel("case (sorted)")
    ax2.set_ylabel("volume (mL)")
    ax2.set_xlim(-1, len(pc))
    ax2.set_xticks([0, len(pc) - 1])
    ax2.set_xticklabels([pc[0]["case"].replace("case_", ""), pc[-1]["case"].replace("case_", "")])
    ax2.legend(loc="upper left", ncol=3)

    for name, i in (("smallest", 0), ("largest", len(pc) - 1)):
        ax2.annotate(
            f"{pc[i]['case']}\n{vol[i]:.0f} mL",
            xy=(i, bottom[i]), xytext=(0, 8), textcoords="offset points",
            ha="left" if i == 0 else "right", fontsize=8, color=S.INK, fontweight="bold",
        )

    S.titled(
        fig, "F01",
        f"A {hl['burden_ratio_max_min']:.0f}x spread in tumour burden across the cohort. "
        f"All three sub-labels are present in every case.",
    )
    S.footnote(fig, f"n = {hl['n_cases']} cases · 1 mm isotropic, so 1 voxel = 1 mm³ and 1000 voxels = 1 mL.")
    S.save(fig, "F01")


def fig02_normalisation(stats: dict, hl: dict) -> None:
    import matplotlib.pyplot as plt

    pc = stats["per_case"]
    raw_x = (RAW_BINS[:-1] + RAW_BINS[1:]) / 2
    nrm_x = (NRM_BINS[:-1] + NRM_BINS[1:]) / 2

    flair = [c["brain_mean"][0] for c in pc]
    lo_case = pc[int(np.argmin(flair))]["case"]
    hi_case = pc[int(np.argmax(flair))]["case"]
    marked = {lo_case: S.SERIES[1], hi_case: S.SERIES[0]}

    fig, axes = plt.subplots(C.N_CHANNELS, 2, figsize=(9.5, 8.4), sharex="col")
    fig.subplots_adjust(hspace=0.30, wspace=0.10)

    def smooth(y):
        """The source volumes are integer-valued, so after dividing by a small
        per-case standard deviation the z-scores land on a coarse lattice and the
        histogram combs. A 5-point moving average removes the sampling artefact
        without moving the distribution."""
        k = np.ones(5) / 5
        return np.convolve(np.asarray(y, dtype=float), k, mode="same")

    for r, mod in enumerate(C.MODALITIES):
        for col, (key, xs, label, xlim) in enumerate(
            [
                ("raw_hist", raw_x, "raw intensity (a.u.)", (0, 1500)),
                ("nrm_hist", nrm_x, "z-score within brain", (-3, 5)),
            ]
        ):
            ax = axes[r, col]
            for c in pc:
                if c["case"] not in marked:
                    ax.plot(xs, smooth(c[key][r]), color=S.MUTED, alpha=0.28, linewidth=0.7)
            for cid, colour in marked.items():
                c = next(k for k in pc if k["case"] == cid)
                ax.plot(xs, smooth(c[key][r]), color=colour, linewidth=2.0, zorder=5)
            ax.set_xlim(*xlim)
            ax.set_yticks([])
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(mod, fontweight="bold", color=S.INK)
            if r == C.N_CHANNELS - 1:
                ax.set_xlabel(label)
            if r == 0:
                ax.set_title(("a · Before — as stored" if col == 0 else "b · After — per-case z-score"))

    for cid, colour in marked.items():
        axes[0, 0].plot([], [], color=colour, linewidth=2.0, label=cid)
    axes[0, 0].plot([], [], color=S.MUTED, alpha=0.5, linewidth=0.9, label="other 58 cases")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    lo, hi = hl["flair_brain_mean_range"]
    S.titled(
        fig, "F02",
        f"Brain-mean FLAIR spans {lo:.0f} to {hi:.0f} across cases — a {hl['flair_brain_mean_ratio']:.1f}x "
        f"spread with no clinical meaning. Fed raw, a model learns which case it is looking at.",
    )
    S.footnote(
        fig,
        f"n = {hl['n_cases']} cases, one curve each · brain voxels only (skull-stripped background excluded) · "
        "densities from 160-bin histograms, 5-point smoothed · x-axes clipped to the populated range.",
    )
    S.save(fig, "F02")


def fig03_example(case_id: str) -> None:
    import matplotlib.pyplot as plt

    images, labels, brain = D.load_cached(case_id, mmap=False)
    tumour = labels > 0
    z = int(tumour.sum(axis=(1, 2)).argmax())

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.4))
    fig.subplots_adjust(hspace=0.02, wspace=0.03)

    for i, ax in enumerate(axes.ravel()):
        disp = D.window_to_uint8(images[i].astype(np.float32), brain)
        ax.imshow(disp[z], cmap="gray", vmin=0, vmax=255)
        ax.contour(tumour[z], levels=[0.5], colors=[S.CONTOUR], linewidths=1.4)
        S.bare(ax)
        ax.text(
            0.03, 0.96, C.MODALITIES[i], transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold", color="white",
        )

    S.titled(
        fig, "F03",
        f"{case_id}, axial slice {z} — the slice with the largest lesion cross-section. "
        "The expert mask (outline) is one label shared by all four channels.",
    )
    S.footnote(
        fig,
        f"Whole-lesion contour = any label > 0 · display window = "
        f"{C.WINDOW_PCT[0]}–{C.WINDOW_PCT[1]} percentile of brain voxels, per channel.",
    )
    S.save(fig, "F03")


def fig04_spatial(hl: dict) -> None:
    import matplotlib.pyplot as plt

    prior = np.load(C.OUTPUTS / "spatial_prior.npy")  # (z, y, x)
    slice_area = np.load(C.OUTPUTS / "slice_area.npy")

    fig = plt.figure(figsize=(9.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.30, wspace=0.12)

    views = [
        ("axial", prior.mean(axis=0)),
        ("coronal", prior.mean(axis=1)),
        ("sagittal", prior.mean(axis=2)),
    ]
    vmax = max(v.max() for _, v in views)
    for i, (name, proj) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(proj, cmap=S.SEQUENTIAL, vmin=0, vmax=vmax, aspect="equal")
        S.bare(ax)
        ax.set_title(f"{'abc'[i]} · {name}", fontsize=9.5)
        if i == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("fraction of cases", fontsize=7.5, color=S.INK_SOFT)
            cb.ax.tick_params(labelsize=7, colors=S.MUTED)
            cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    z = np.arange(slice_area.shape[1])
    for row in slice_area:
        ax.plot(z, row / 1000.0, color=S.MUTED, alpha=0.25, linewidth=0.7)
    ax.plot(z, slice_area.mean(axis=0) / 1000.0, color=S.SERIES[0], linewidth=2.4, label="cohort mean", zorder=5)
    ax.set_title("d · Tumour cross-section by axial slice")
    ax.set_xlabel("axial slice index (inferior → superior)")
    ax.set_ylabel("area (10³ px)")
    ax.set_xlim(0, slice_area.shape[1] - 1)
    ax.legend(loc="upper right")

    S.titled(
        fig, "F04",
        "Lesions concentrate supratentorially and off-midline, but the cohort covers both hemispheres. "
        "No single slice range can be assumed to contain the tumour.",
    )
    S.footnote(
        fig,
        f"n = {hl['n_cases']} cases · a–c: mean of the binary whole-lesion masks, projected along each axis, "
        "in cropped space.",
    )
    S.save(fig, "F04")


def fig05_imbalance(stats: dict, hl: dict) -> None:
    import matplotlib.pyplot as plt

    pc = stats["per_case"]
    total = float(np.prod(C.VOLUME_SHAPE))
    rows = sorted(
        (
            {
                "case": c["case"],
                "vol": 100 * c["tumour_voxels"] / total,
                "brain": 100 * c["tumour_voxels"] / c["brain_voxels"],
            }
            for c in pc
        ),
        key=lambda r: r["brain"],
    )
    y = np.arange(len(rows))
    a = np.array([r["vol"] for r in rows])
    b = np.array([r["brain"] for r in rows])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 6.6), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.28}
    )

    ax.hlines(y, a, b, color=S.GRID, linewidth=1.6, zorder=1)
    ax.scatter(a, y, s=22, color=S.SERIES[0], zorder=3, label="of whole volume")
    ax.scatter(b, y, s=22, color=S.SERIES[1], zorder=3, label="of brain only")
    ax.set_title("a · Tumour as a share of what you count")
    ax.set_xlabel("tumour voxels (% of denominator)")
    ax.set_ylabel("case (sorted)")
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    ax.annotate(
        f"median {np.median(a):.2f}%",
        xy=(np.median(a), len(rows) * 0.5), xytext=(-8, 0), textcoords="offset points",
        ha="right", fontsize=8.5, fontweight="bold", color=S.SERIES[0],
    )
    ax.annotate(
        f"median {np.median(b):.1f}%",
        xy=(np.median(b), len(rows) * 0.5), xytext=(8, 0), textcoords="offset points",
        ha="left", fontsize=8.5, fontweight="bold", color=S.SERIES[1],
    )

    slice_area = np.load(C.OUTPUTS / "slice_area.npy")
    frac_pos = 100 * (slice_area > 0).mean()
    bars = [
        ("voxels,\nwhole volume", float(np.median(a)), S.SERIES[0]),
        ("voxels,\nbrain only", float(np.median(b)), S.SERIES[1]),
        ("axial slices\ncontaining tumour", float(frac_pos), S.SERIES[2]),
    ]
    ax2.bar([b_[0] for b_ in bars], [b_[1] for b_ in bars], color=[b_[2] for b_ in bars], width=0.66)
    for i, (_, v, _) in enumerate(bars):
        ax2.text(i, v + 1.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color=S.INK)
    ax2.set_title("b · Positive rate by unit")
    ax2.set_ylabel("% positive")
    ax2.set_ylim(0, 56)
    ax2.tick_params(axis="x", labelsize=7.5)

    S.titled(
        fig, "F05",
        f"The imbalance depends entirely on the denominator: {np.median(a):.2f}% of the volume, "
        f"{np.median(b):.1f}% of the brain, {frac_pos:.0f}% of axial slices. Crop to brain, and train on slices.",
    )
    S.footnote(
        fig,
        f"n = {hl['n_cases']} cases · brain = any voxel non-zero in any channel · "
        "background air is a free true-negative and is excluded from evaluation.",
    )
    S.save(fig, "F05")


# ── verification ───────────────────────────────────────────────────────────

EXPECTED = {
    "tumour_voxels.median": (108705, 0),
    "tumour_voxels.min": (7285, 0),
    "tumour_voxels.max": (256875, 0),
    "positive_slices": (4290, 0),
    "total_slices": (9300, 0),
    "brain_fraction_mean": (0.15969, 1e-5),
    "tumour_frac_volume_median": (0.012176, 1e-5),
    "tiny_positive_slice_fraction": (0.093, 5e-4),
}


def verify(hl: dict) -> int:
    failed = 0
    for key, (want, tol) in EXPECTED.items():
        got = hl
        for part in key.split("."):
            got = got[part]
        ok = abs(got - want) <= tol
        print(f"  {'PASS' if ok else 'FAIL'}  {key}: {got} (expected {want})")
        failed += not ok
    return failed


# ── entry point ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute stats from the raw volumes")
    ap.add_argument("--tier", default="standard", choices=["standard", "tiny"])
    ap.add_argument("--example", default="case_001", help="case to draw for F03")
    args = ap.parse_args(argv)

    S.apply()
    ids = C.case_ids(args.tier)
    t0 = time.time()

    stats = load_stats(ids, force=args.force)
    hl = headline(stats)
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    (C.OUTPUTS / "eda_headline.json").write_text(json.dumps(hl, indent=2))

    print(f"\nheadline numbers (n = {hl['n_cases']}):")
    tv = hl["tumour_voxels"]
    print(f"  tumour voxels   min {tv['min']}  p25 {tv['p25']}  median {tv['median']}  "
          f"p75 {tv['p75']}  max {tv['max']}  ({hl['burden_ratio_max_min']:.0f}x spread)")
    print(f"  brain fraction  {hl['brain_fraction_mean']:.4f} of the volume")
    print(f"  tumour is       {100 * hl['tumour_frac_volume_median']:.2f}% of volume, "
          f"{100 * hl['tumour_frac_brain_median']:.1f}% of brain (medians)")
    print(f"  positive slices {hl['positive_slices']}/{hl['total_slices']} "
          f"({100 * hl['positive_slice_fraction']:.1f}%)")
    print(f"  of those, {100 * hl['tiny_positive_slice_fraction']:.1f}% carry <100 px of tumour")
    print(f"  FLAIR brain-mean {hl['flair_brain_mean_range'][0]:.0f}..{hl['flair_brain_mean_range'][1]:.0f} "
          f"({hl['flair_brain_mean_ratio']:.1f}x spread)")

    if args.verify:
        print("\nverification against pre-recorded values:")
        failed = verify(hl)
        if failed:
            print(f"\n{failed} check(s) FAILED")
            return 1

    print("\ndrawing figures:")
    for name, fn in [
        ("F01", lambda: fig01_cohort(stats, hl)),
        ("F02", lambda: fig02_normalisation(stats, hl)),
        ("F03", lambda: fig03_example(args.example)),
        ("F04", lambda: fig04_spatial(hl)),
        ("F05", lambda: fig05_imbalance(stats, hl)),
    ]:
        fn()
        print(f"  {name}  {C.FIGURES / (name + '.png')}")

    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
