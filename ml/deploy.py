"""Compare every trained arm, install the best one, and publish it to the website.

An "arm" is one completed training+inference run: a directory containing an
`infer_<arch>.json` (the honest out-of-fold score), a `checkpoints/` directory,
and the per-case `predictions/`. The pre-registered run and each overnight stage
are all arms.

    python -m ml.deploy --report            # compare arms, change nothing
    python -m ml.deploy --auto              # compare, install the best, export web
    python -m ml.deploy --install <tag>     # install a specific arm instead

**Deploying is not reporting.** The number this project reports against the
RULES.md targets is the pre-registered arm's, fixed on 2026-08-17 and archived in
`outputs/prereg_20260817/`. Selecting the best arm afterwards is model selection
for *deployment and for the unseen test set*, which is a different decision with
different rules — it is allowed to use everything we now know. Publishing a
better arm to the site does NOT retroactively become the pre-registered result,
and `deploy_manifest.json` records both so the distinction cannot quietly erode.

Selection uses mean out-of-fold Dice — the same metric the targets are stated in.
It deliberately does NOT use the patch proxy, which is noisy, optimistic, and
already caused one real defect (see outputs/prereg_20260817/README.md).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

from . import config as C

PREREG = C.OUTPUTS / "prereg_20260817"
OVERNIGHT = C.OUTPUTS / "overnight"
MANIFEST = C.OUTPUTS / "deploy_manifest.json"


def _read_arm(d: pathlib.Path, tag: str, *, prereg: bool = False) -> list[dict]:
    """An arm directory may hold a 3d result, a 25d result, or both."""
    arms = []
    for arch in ("3d", "25d"):
        p = d / f"infer_{arch}.json"
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text())["summary"]
        except Exception as e:
            print(f"  ! {tag}/{p.name} unreadable: {type(e).__name__}: {e}")
            continue
        # Checkpoints may sit in <arm>/checkpoints/ (how overnight.sh archives
        # them) or directly in <arm>/ (how the pre-registered run was archived).
        # Accept both, or the pre-registered arm cannot serve as the fallback.
        ck = sorted((d / "checkpoints").glob(f"{arch}_fold*.pt")) if (d / "checkpoints").is_dir() else []
        if not ck:
            ck = sorted(d.glob(f"{arch}_fold*.pt"))
        ck = [c for c in ck if not c.name.endswith("_final.pt")]
        arms.append({
            "tag": tag, "arch": arch, "dir": str(d), "prereg": prereg,
            "mean_dice": s.get("mean_dice"), "median_dice": s.get("median_dice"),
            "min_dice": s.get("min_dice"), "ci95": s.get("mean_dice_ci95"),
            "n": s.get("n"), "tta": s.get("tta"),
            "n_checkpoints": len(ck), "complete": len(ck) == C.N_FOLDS,
        })
    return arms


def discover() -> list[dict]:
    arms = _read_arm(PREREG, "prereg_20260817", prereg=True)
    if OVERNIGHT.is_dir():
        for d in sorted(OVERNIGHT.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                arms.extend(_read_arm(d, d.name))
    # Anything still sitting in the working paths, i.e. a stage that finished
    # inference but whose archive step has not run yet.
    arms.extend(_read_arm(C.OUTPUTS, "outputs (unarchived)"))
    return arms


def report(arms: list[dict]) -> None:
    if not arms:
        print("no arms found — nothing has completed training + inference yet")
        return
    print(f"{'tag':26s} {'arch':5s} {'mean':>7s} {'median':>7s} {'worst':>7s} "
          f"{'n':>3s} {'ckpt':>5s} {'note':>10s}")
    for a in sorted(arms, key=lambda x: -(x["mean_dice"] or 0)):
        note = "PRE-REG" if a["prereg"] else ("" if a["complete"] else "INCOMPLETE")
        md = a["mean_dice"]
        print(f"{a['tag'][:26]:26s} {a['arch']:5s} "
              f"{md:>7.4f} {a['median_dice']:>7.4f} {a['min_dice']:>7.4f} "
              f"{a['n']:>3d} {a['n_checkpoints']:>5d} {note:>10s}")

    T = C.TARGETS
    print(f"\ntargets: mean>={T['mean_dice_floor']} median>={T['median_dice_floor']} "
          f"worst>={T['worst_dice_floor']}  (stretch mean/median {T['mean_dice_stretch']})")
    pre = next((a for a in arms if a["prereg"] and a["arch"] == "3d"), None)
    if pre:
        print(f"\nREPORTED (pre-registered, immutable): mean {pre['mean_dice']:.4f} "
              f"median {pre['median_dice']:.4f} worst {pre['min_dice']:.4f}")
    best = pick(arms)
    if best:
        print(f"DEPLOY CANDIDATE (best available): {best['tag']} / {best['arch']} "
              f"mean {best['mean_dice']:.4f}")
        if pre and best["mean_dice"] is not None:
            print(f"  delta vs pre-registered: {best['mean_dice'] - pre['mean_dice']:+.4f} mean Dice")


def pick(arms: list[dict]) -> dict | None:
    """Best complete arm by mean out-of-fold Dice."""
    usable = [a for a in arms if a["complete"] and a["mean_dice"] is not None]
    return max(usable, key=lambda a: a["mean_dice"]) if usable else None


def install(arm: dict, *, export: bool = True) -> dict:
    """Put an arm's checkpoints, scores and predictions at the canonical paths."""
    d, arch = pathlib.Path(arm["dir"]), arm["arch"]
    ckpt_out = C.OUTPUTS / "checkpoints"
    ckpt_out.mkdir(parents=True, exist_ok=True)

    srcs = sorted((d / "checkpoints").glob(f"{arch}_fold*.pt"))
    if not srcs:
        srcs = sorted(d.glob(f"{arch}_fold*.pt"))
    installed = []
    for src in srcs:
        if src.name.endswith("_final.pt"):
            continue
        if src.resolve() != (ckpt_out / src.name).resolve():
            shutil.copy2(src, ckpt_out / src.name)
        installed.append(src.name)

    if d != C.OUTPUTS:
        src_json = d / f"infer_{arch}.json"
        if src_json.exists():
            shutil.copy2(src_json, C.OUTPUTS / f"infer_{arch}.json")
        src_hist = d / f"train_history_{arch}.json"
        if src_hist.exists():
            shutil.copy2(src_hist, C.OUTPUTS / f"train_history_{arch}.json")
        src_pred = d / "predictions" / arch
        if src_pred.is_dir():
            dst = C.OUTPUTS / "predictions" / arch
            dst.mkdir(parents=True, exist_ok=True)
            for f in src_pred.glob("*.npz"):
                shutil.copy2(f, dst / f.name)

    print(f"installed {len(installed)} checkpoints from {arm['tag']} ({arch}) -> {ckpt_out}")

    web_ok, web_err = None, None
    if export:
        print(f"exporting web assets (--full --arch {arch}) ...")
        r = subprocess.run([sys.executable, "-m", "ml.export_web", "--full", "--arch", arch],
                           capture_output=True, text=True)
        web_ok = r.returncode == 0
        if not web_ok:
            web_err = (r.stderr or r.stdout)[-2000:]
            print(f"  web export FAILED:\n{web_err}")
        else:
            print(f"  web export ok -> {C.WEB_CASES}")

    pre = next((a for a in discover() if a["prereg"] and a["arch"] == "3d"), None)
    manifest = {
        "deployed": {k: arm[k] for k in ("tag", "arch", "dir", "mean_dice", "median_dice",
                                         "min_dice", "ci95", "n")},
        "checkpoints_installed": installed,
        "web_export_ok": web_ok,
        "web_export_error": web_err,
        # Recorded together and labelled, so "what we deployed" can never be
        # mistaken for "what we pre-registered and reported".
        "reported_prereg": None if pre is None else {
            k: pre[k] for k in ("tag", "arch", "mean_dice", "median_dice", "min_dice", "ci95", "n")},
        "note": ("Deployment/test-set model selection. The REPORTED result against RULES.md "
                 "targets is `reported_prereg` and is not changed by deploying a better arm."),
        "predict_command": ("python3 -m ml.predict --input-dir <TESTSET_DIR> "
                            "--output-dir outputs/testset"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", action="store_true", help="compare arms, change nothing")
    g.add_argument("--auto", action="store_true", help="install the best arm and export the site")
    g.add_argument("--install", metavar="TAG", help="install a named arm instead of the best")
    ap.add_argument("--arch", default=None, choices=["25d", "3d"], help="restrict to one architecture")
    ap.add_argument("--no-export", action="store_true", help="install checkpoints but skip the web export")
    args = ap.parse_args(argv)

    arms = discover()
    if args.arch:
        arms = [a for a in arms if a["arch"] == args.arch]

    if args.report:
        report(arms)
        return 0

    if args.install:
        cand = [a for a in arms if a["tag"] == args.install]
        if not cand:
            print(f"no arm tagged {args.install!r}. available: {sorted({a['tag'] for a in arms})}")
            return 1
        arm = max(cand, key=lambda a: a["mean_dice"] or 0)
    else:
        arm = pick(arms)
        if arm is None:
            print("no complete arm to deploy — leaving the canonical checkpoints untouched")
            report(arms)
            return 1

    report(arms)
    print()
    install(arm, export=not args.no_export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
