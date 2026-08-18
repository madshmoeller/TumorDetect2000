"""Figures F06-F16: architecture, training, and results.

F01-F05 live in eda.py (Act I, ground truth only). These depend on training
and inference having produced outputs/train_history_*.json, outputs/infer_*.json
and outputs/baselines_summary.json — run baselines.py, train.py and infer.py
first.

    python -m ml.figures
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import config as C
from . import figstyle as S
from . import metrics as M

ARCH_LABEL = {"25d": "2.5D U-Net", "3d": "3D U-Net"}


def _load(name: str) -> dict | None:
    path = C.OUTPUTS / name
    return json.loads(path.read_text()) if path.exists() else None


# ── F07: training curves ────────────────────────────────────────────────────


def fig07_training_curves() -> bool:
    import matplotlib.pyplot as plt

    histories = {a: _load(f"train_history_{a}.json") for a in ("25d", "3d")}
    histories = {a: h for a, h in histories.items() if h}
    if not histories:
        return False

    fig, axes = plt.subplots(2, len(histories), figsize=(5.2 * len(histories), 6.4), squeeze=False)

    for col, (arch, h) in enumerate(histories.items()):
        ax_loss, ax_dice = axes[0, col], axes[1, col]
        for fold_result in h["folds"]:
            f = fold_result["fold"]
            epochs = [e["epoch"] for e in fold_result["history"]]
            loss = [e["train_loss"] for e in fold_result["history"]]
            dice = [e["val_dice"] for e in fold_result["history"]]
            colour = S.SERIES[f % len(S.SERIES)] if f < len(S.SERIES) else S.MUTED
            ax_loss.plot(epochs, loss, color=colour, alpha=0.85, linewidth=1.4, label=f"fold {f}")
            ax_dice.plot(epochs, dice, color=colour, alpha=0.85, linewidth=1.4)
            best_e = fold_result["best_epoch"]
            ax_dice.scatter([best_e], [fold_result["best_val_dice"]], color=colour, s=22, zorder=5)

        ax_loss.set_title(f"{ARCH_LABEL[arch]} — training loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("Dice + BCE loss")
        if col == 0:
            ax_loss.legend(loc="upper right", fontsize=7.5, ncol=2)

        ax_dice.set_title("validation Dice (patch/slice proxy, not the reported metric)")
        ax_dice.set_xlabel("epoch")
        ax_dice.set_ylabel("Dice")
        ax_dice.set_ylim(0, 1)

    S.titled(
        fig, "F07",
        "Dots mark each fold's early-stopping checkpoint. The curve here is a fast per-batch proxy for "
        "monitoring only — the reported Dice (F09) is computed once, out-of-fold, on full reconstructed volumes.",
    )
    S.footnote(fig, f"{C.N_FOLDS}-fold CV · early stop patience {C.EARLY_STOP_PATIENCE} epochs, no val-Dice improvement.")
    S.save(fig, "F07")
    return True


# ── F08: threshold sweep ────────────────────────────────────────────────────


def fig08_threshold_sweep() -> bool:
    import matplotlib.pyplot as plt

    infers = {a: _load(f"infer_{a}.json") for a in ("25d", "3d")}
    infers = {a: v for a, v in infers.items() if v and v["summary"].get("threshold_sweep")}
    if not infers:
        return False

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for i, (arch, v) in enumerate(infers.items()):
        # JSON round-trips dict keys to strings; re-key by float rather than
        # relying on str(t) reproducing the exact original formatting.
        sweep = {float(t): v for t, v in v["summary"]["threshold_sweep"].items()}
        ts = sorted(sweep)
        dices = [sweep[t] for t in ts]
        colour = S.SERIES[i]
        ax.plot(ts, dices, color=colour, linewidth=2.2, marker="o", markersize=4, label=ARCH_LABEL[arch])
        best_i = int(np.argmax(dices))
        ax.scatter([ts[best_i]], [dices[best_i]], color=colour, s=90, zorder=6,
                   edgecolor=S.INK, linewidth=1.2)
        ax.annotate(f"{ts[best_i]:.2f}", xy=(ts[best_i], dices[best_i]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color=colour)

    ax.axvline(C.DEFAULT_THRESHOLD, color=S.MUTED, linestyle="--", linewidth=1, zorder=1)
    ax.annotate("shipped default", xy=(C.DEFAULT_THRESHOLD, ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=S.MUTED)
    ax.set_xlabel("decision threshold (sigmoid probability)")
    ax.set_ylabel("mean Dice, out-of-fold")
    ax.legend(loc="lower center")

    S.titled(
        fig, "F08",
        "Mean out-of-fold Dice as the decision threshold varies. Swept without post-processing — "
        "the connected-component step is a separate ablation (F13).",
    )
    S.footnote(fig, "n = 60, all cases, no post-processing applied before thresholding.")
    S.save(fig, "F08")
    return True


# ── shared data plumbing ─────────────────────────────────────────────────────


def _model_series() -> dict:
    """Every scored model/baseline, as {label: {case_id: dice}}, in report order."""
    series = {}
    baselines = _load("baselines.json")
    if baselines:
        series["FLAIR threshold"] = {c: v["dice"] for c, v in baselines["flair_threshold"].items()}
        series["Random forest"] = {c: v["dice"] for c, v in baselines["random_forest"].items()}
    for arch in ("25d", "3d"):
        inf = _load(f"infer_{arch}.json")
        if inf:
            series[ARCH_LABEL[arch]] = {c: v["dice"] for c, v in inf["per_case"].items()}
    return series


def _primary_arch() -> str | None:
    """3D is the primary architecture (see docs/model_scope.md); fall back to 2.5D if it's missing."""
    if _load("infer_3d.json"):
        return "3d"
    if _load("infer_25d.json"):
        return "25d"
    return None


# ── F09: the headline ────────────────────────────────────────────────────────


def fig09_headline() -> bool:
    import matplotlib.pyplot as plt

    series = _model_series()
    if len(series) < 2:
        return False

    names = list(series.keys())
    primary = ARCH_LABEL.get(_primary_arch())
    common_cases = sorted(set.intersection(*(set(d) for d in series.values())))

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    positions = np.arange(len(names))
    colours = [S.SERIES[i % len(S.SERIES)] for i in range(len(names))]

    for i, name in enumerate(names):
        vals = np.array([series[name][c] for c in common_cases])
        parts = ax.violinplot([vals], positions=[i], showextrema=False, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor(colours[i])
            body.set_alpha(0.28)
        jitter = (np.random.RandomState(i).rand(len(vals)) - 0.5) * 0.18
        ax.scatter(i + jitter, vals, s=14, color=colours[i], alpha=0.75, zorder=3)
        ax.hlines(np.median(vals), i - 0.32, i + 0.32, color=S.INK, linewidth=2.2, zorder=4)

    if primary and primary in series:
        strongest_baseline = next((n for n in names if n in ("Random forest", "FLAIR threshold")), None)
        if strongest_baseline:
            a = np.array([series[primary][c] for c in common_cases])
            b = np.array([series[strongest_baseline][c] for c in common_cases])
            p = M.paired_wilcoxon(a, b)
            ax.annotate(
                f"{primary} vs {strongest_baseline}: paired Wilcoxon p = {p:.1e}",
                xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=9, color=S.INK_SOFT,
            )

    for t, label in (("mean_dice_floor", "floor"), ("mean_dice_stretch", "stretch")):
        ax.axhline(C.TARGETS[t], color=S.MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.annotate(f"{label} {C.TARGETS[t]:.2f}", xy=(len(names) - 0.5, C.TARGETS[t]), xytext=(4, 2),
                    textcoords="offset points", fontsize=7.5, color=S.MUTED)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Dice, out-of-fold")
    ax.set_ylim(0, 1.02)

    S.titled(
        fig, "F09",
        f"Per-case Dice, {len(common_cases)} cases scored by every model shown. "
        "Dashed lines mark the pre-registered floor/stretch targets (RULES.md), fixed before training.",
    )
    S.footnote(fig, "Thick line = median. Points = individual cases, jittered horizontally for visibility.")
    S.save(fig, "F09")
    return True


# ── F10: Dice vs tumour size ─────────────────────────────────────────────────


def fig10_dice_vs_size() -> bool:
    import matplotlib.pyplot as plt

    arch = _primary_arch()
    inf = _load(f"infer_{arch}.json") if arch else None
    if not inf:
        return False

    cases = inf["per_case"]
    vol = np.array([v["true_volume_mm3"] / 1000 for v in cases.values()])  # mL
    dice = np.array([v["dice"] for v in cases.values()])

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(vol, dice, s=26, color=S.SERIES[0], alpha=0.8, edgecolor=S.PAPER, linewidth=0.4)

    order = np.argsort(vol)
    if len(vol) >= 8:
        window = max(5, len(vol) // 6)
        kernel = np.ones(window) / window
        smoothed = np.convolve(dice[order], kernel, mode="valid")
        x_s = vol[order][window // 2 : window // 2 + len(smoothed)]
        ax.plot(x_s, smoothed, color=S.INK, linewidth=2.0, label=f"rolling mean (window={window})")
        ax.legend(loc="lower right")

    r = float(np.corrcoef(vol, dice)[0, 1])
    ax.annotate(f"Pearson r = {r:.2f}", xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9, color=S.INK_SOFT)
    ax.set_xlabel("true tumour volume (mL)")
    ax.set_ylabel("Dice, out-of-fold")
    ax.set_ylim(0, 1.02)

    S.titled(
        fig, "F10",
        f"{ARCH_LABEL[arch]}: does a smaller lesion predict a harder case? "
        f"(F01 showed a {35}x spread in tumour burden across this cohort.)",
    )
    S.footnote(fig, f"n = {len(vol)} cases, out-of-fold predictions, {ARCH_LABEL[arch]}.")
    S.save(fig, "F10")
    return True


# ── F11: qualitative — best / median / worst ─────────────────────────────────


def fig11_qualitative() -> bool:
    import matplotlib.pyplot as plt

    from . import data as D

    arch = _primary_arch()
    inf = _load(f"infer_{arch}.json") if arch else None
    if not inf:
        return False

    cases = inf["per_case"]
    ranked = sorted(cases.items(), key=lambda kv: kv[1]["dice"])
    picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    for ax, (label, (cid, m)) in zip(axes, picks.items()):
        images, labels, brain = D.load_cached(cid, mmap=False)
        pred_path = C.OUTPUTS / "predictions" / arch / f"{cid}.npz"
        pred = np.load(pred_path)["pred"].astype(bool)
        gt = np.asarray(labels) > 0

        overlap = (pred.astype(int) + gt.astype(int)).sum(axis=(1, 2))
        z = int(np.argmax(overlap)) if overlap.any() else pred.shape[0] // 2

        flair = D.window_to_uint8(np.asarray(images[0, z]).astype(np.float32), np.asarray(brain[z]))
        ax.imshow(flair, cmap="gray", vmin=0, vmax=255)

        tp = pred[z] & gt[z]
        fn = ~pred[z] & gt[z]
        fp = pred[z] & ~gt[z]
        for arr, colour in ((tp, S.DIFF["correct"]), (fn, S.DIFF["missed"]), (fp, S.DIFF["imagined"])):
            ax.imshow(np.ma.masked_where(~arr, arr), cmap=S.mask_cmap(colour, alpha=0.75), vmin=0, vmax=1)

        S.bare(ax)
        ax.set_title(f"{label} · {cid} · Dice {m['dice']:.2f}", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=S.DIFF[k], markersize=10, label=S.DIFF_LABEL[k])
        for k in ("correct", "missed", "imagined")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)

    S.titled(fig, "F11", f"{ARCH_LABEL[arch]}: best, median and worst out-of-fold cases by Dice.")
    S.footnote(fig, "Slice shown = the axial slice with the largest prediction/ground-truth overlap for that case.")
    S.save(fig, "F11")
    return True


# ── F12: volume agreement ────────────────────────────────────────────────────


def fig12_volume_agreement() -> bool:
    import matplotlib.pyplot as plt

    arch = _primary_arch()
    inf = _load(f"infer_{arch}.json") if arch else None
    if not inf:
        return False

    cases = inf["per_case"]
    true_v = np.array([v["true_volume_mm3"] / 1000 for v in cases.values()])
    pred_v = np.array([v["pred_volume_mm3"] / 1000 for v in cases.values()])
    mean_v = (true_v + pred_v) / 2
    diff_v = pred_v - true_v

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 5.0))

    lims = [0, max(true_v.max(), pred_v.max()) * 1.05]
    ax1.plot(lims, lims, color=S.MUTED, linewidth=1, linestyle="--", zorder=1)
    ax1.scatter(true_v, pred_v, s=24, color=S.SERIES[0], alpha=0.8, edgecolor=S.PAPER, linewidth=0.4, zorder=3)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("true volume (mL)")
    ax1.set_ylabel("predicted volume (mL)")
    ax1.set_title("a · Agreement")

    bias, sd = float(diff_v.mean()), float(diff_v.std())
    ax2.scatter(mean_v, diff_v, s=24, color=S.SERIES[0], alpha=0.8, edgecolor=S.PAPER, linewidth=0.4, zorder=3)
    ax2.axhline(bias, color=S.INK, linewidth=1.6, zorder=2)
    for k, ls in ((1.96, "--"), (-1.96, "--")):
        ax2.axhline(bias + k * sd, color=S.MUTED, linewidth=1, linestyle=ls, zorder=1)
    ax2.annotate(f"bias {bias:+.1f} mL", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=8.5, color=S.INK)
    ax2.set_xlabel("mean of true & predicted (mL)")
    ax2.set_ylabel("predicted − true (mL)")
    ax2.set_title("b · Bland–Altman")

    S.titled(fig, "F12", f"{ARCH_LABEL[arch]}: predicted vs. true tumour volume, out-of-fold.")
    S.footnote(fig, f"n = {len(true_v)} cases · dashed lines in (b) = bias ± 1.96 SD.")
    S.save(fig, "F12")
    return True


# ── F13: ablations ───────────────────────────────────────────────────────────


def fig13_ablations() -> bool:
    import matplotlib.pyplot as plt

    bars = []  # (label, dice_array)
    baselines = _load("baselines.json")
    if baselines:
        bars.append(("FLAIR\nthreshold", np.array([v["dice"] for v in baselines["flair_threshold"].values()])))
        bars.append(("Random\nforest", np.array([v["dice"] for v in baselines["random_forest"].values()])))

    for arch in ("25d", "3d"):
        inf = _load(f"infer_{arch}.json")
        if not inf:
            continue
        bars.append((f"{ARCH_LABEL[arch]}\n(shipped)", np.array([v["dice"] for v in inf["per_case"].values()])))
        sweep = inf["summary"].get("threshold_sweep")
        if sweep:
            sweep_f = {float(t): v for t, v in sweep.items()}
            if C.DEFAULT_THRESHOLD in sweep_f:
                # Sweep stores mean only; the per-case array is unavailable here,
                # so this bar uses the mean directly with no CI rather than fabricate spread.
                bars.append((f"{ARCH_LABEL[arch]}\nno postproc.", np.array([sweep_f[C.DEFAULT_THRESHOLD]])))

        notta = _load(f"infer_{arch}_notta.json")
        if notta:
            bars.append((f"{ARCH_LABEL[arch]}\nno TTA", np.array([v["dice"] for v in notta["per_case"].values()])))

    if len(bars) < 2:
        return False

    fig, ax = plt.subplots(figsize=(max(7.0, 1.35 * len(bars)), 5.6))
    x = np.arange(len(bars))
    for i, (label, vals) in enumerate(bars):
        mean = float(vals.mean())
        colour = S.SERIES[i % len(S.SERIES)]
        if len(vals) > 1:
            lo, hi = M.bootstrap_ci(vals)
            ax.bar(i, mean, color=colour, width=0.62, zorder=3)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=S.INK, capsize=4, linewidth=1.4, zorder=4)
        else:
            ax.bar(i, mean, color=colour, width=0.62, zorder=3, hatch="//", alpha=0.7)
        ax.annotate(f"{mean:.3f}", xy=(i, mean + 0.015), ha="center", fontsize=8.5, fontweight="bold", color=S.INK)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("mean Dice, out-of-fold")
    ax.set_ylim(0, 1.05)
    ax.axhline(C.TARGETS["mean_dice_floor"], color=S.MUTED, linestyle="--", linewidth=1)

    S.titled(
        fig, "F13",
        "What each design choice is worth. Hatched bars have no error bar (single aggregate figure, not a "
        "per-case array) — see the footnote before reading them as precise as the solid bars.",
    )
    S.footnote(
        fig,
        "Error bars = 95% bootstrap CI on the mean, n=60. 'no postproc.' bars are the mean at threshold=0.5 "
        "with connected-component filtering disabled; a per-case array wasn't retained for those, by design "
        "(ml/infer.py's threshold sweep stores means to avoid caching full probability volumes).",
    )
    S.save(fig, "F13")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args(argv)
    S.apply()

    made = []
    for name, fn in [
        ("F07", fig07_training_curves),
        ("F08", fig08_threshold_sweep),
        ("F09", fig09_headline),
        ("F10", fig10_dice_vs_size),
        ("F11", fig11_qualitative),
        ("F12", fig12_volume_agreement),
        ("F13", fig13_ablations),
    ]:
        if fn():
            made.append(name)
            print(f"  {name}  {C.FIGURES / (name + '.png')}")
        else:
            print(f"  {name}  skipped (missing inputs)")

    print(f"\n{len(made)} figure(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
