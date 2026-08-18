"""One command, start to finish: EDA -> baselines -> train -> infer -> figures -> web export.

Prints a measured wall-clock table at the end — every number in it is a
stopwatch reading taken around the actual call, never an estimate. Compare
that table against the ceilings in RULES.md / config.py, not the other way
around.

    python -m ml.run_all              # the whole pipeline
    python -m ml.run_all --skip-cache-build   # if ml/cache/ is already populated
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import config as C

STEPS: list[tuple[str, list[str]]] = [
    ("preprocess cache", [sys.executable, "-m", "ml.data", "--build"]),
    ("EDA (F01-F05)", [sys.executable, "-m", "ml.eda"]),
    ("baselines", [sys.executable, "-m", "ml.baselines"]),
    ("train 2.5D (5-fold)", [sys.executable, "-m", "ml.train", "--arch", "25d"]),
    ("train 3D (5-fold)", [sys.executable, "-m", "ml.train", "--arch", "3d"]),
    ("infer 2.5D", [sys.executable, "-m", "ml.infer", "--arch", "25d"]),
    ("infer 3D", [sys.executable, "-m", "ml.infer", "--arch", "3d"]),
    ("infer 3D, no TTA (ablation)", [sys.executable, "-m", "ml.infer", "--arch", "3d", "--no-tta"]),
    ("figures (F06-F13)", [sys.executable, "-m", "ml.figures"]),
    ("web export, full", [sys.executable, "-m", "ml.export_web", "--full", "--arch", "3d"]),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-cache-build", action="store_true")
    ap.add_argument("--from-step", type=int, default=0, help="resume from step N (0-indexed), skipping earlier ones")
    args = ap.parse_args(argv)

    steps = STEPS[1:] if args.skip_cache_build and args.from_step == 0 else STEPS[args.from_step:]

    print(f"TumorNet 2000 — full pipeline, {len(steps)} step(s)")
    print(f"ceiling: {C.TRAIN_BUDGET_MINUTES} min training (this machine) / "
          f"{C.MACBOOK_TRAIN_BUDGET_MINUTES} min (M2 Max reference)\n")

    timings = []
    t_total = time.time()
    for name, cmd in steps:
        print(f"── {name} " + "─" * max(0, 60 - len(name)))
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        timings.append((name, elapsed, result.returncode))
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"── {name}: {elapsed / 60:.1f} min  [{status}]\n")
        if result.returncode != 0:
            print("stopping — a step failed. Fix it and resume with --from-step.", file=sys.stderr)
            break

    total = time.time() - t_total
    print("\n" + "=" * 62)
    print("measured wall-clock (not estimated):\n")
    for name, elapsed, code in timings:
        mark = "✓" if code == 0 else "✗"
        print(f"  {mark}  {name:<32} {elapsed / 60:6.1f} min")
    print(f"\n  {'TOTAL':<35} {total / 60:6.1f} min")
    print(f"  vs training ceiling {C.TRAIN_BUDGET_MINUTES} min "
          f"({'within budget' if total / 60 <= C.TRAIN_BUDGET_MINUTES else 'OVER — see RULES.md rule 4'})")
    print("=" * 62)

    return 0 if all(code == 0 for _, _, code in timings) else 1


if __name__ == "__main__":
    sys.exit(main())
