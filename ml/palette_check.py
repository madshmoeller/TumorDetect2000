"""Python port of the data-viz palette validator (the six computable checks).

There is no node runtime on this machine, so the shipped `validate_palette.js`
cannot run. The math here is a line-for-line port: OKLab conversion, the
Machado-Oliveira-Fernandes (2009) CVD transforms at severity 1.0, and the same
thresholds. Ported so the palette checks are *computed*, not eyeballed.

    python -m ml.palette_check "#2a78d6,#eb6834,#1baf7a" --surface "#fdf8ec" --pairs all
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}


def _srgb(h: str) -> tuple[float, float, float]:
    h = h.strip().lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _lin(h: str) -> tuple[float, float, float]:
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in _srgb(h))


def _rel_lum(h: str) -> float:
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_lin(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(h: str) -> tuple[float, float]:
    L, a, b = _oklab_from_lin(_lin(h))
    return L, math.hypot(a, b)


def _simulate(h: str, kind: str):
    r, g, b = _lin(h)
    M = MACHADO[kind]
    return tuple(min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3))


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def validate(palette: list[str], *, mode: str = "light", surface: str | None = None, pairs: str = "adjacent") -> dict:
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    idx = list(range(len(palette)))
    pairlist = list(itertools.combinations(idx, 2)) if pairs == "all" else list(zip(idx, idx[1:]))

    rows, failed = [], 0
    for i, h in enumerate(palette, 1):
        L, Cc = oklch(h)
        ratio = contrast(h, surface)
        band_ok, chroma_ok = lo <= L <= hi, Cc >= CHROMA_FLOOR
        status = "PASS" if band_ok and chroma_ok else "FAIL"
        failed += status == "FAIL"
        rows.append(
            {
                "slot": i,
                "hex": h,
                "L": round(L, 3),
                "C": round(Cc, 3),
                "contrast": round(ratio, 2),
                "band": "ok" if band_ok else f"OUT [{lo},{hi}]",
                "chroma": "ok" if chroma_ok else f"LOW <{CHROMA_FLOOR}",
                "relief": "" if ratio >= CONTRAST_MIN else "WARN <3:1 — needs labels/table",
                "status": status,
            }
        )

    pair_rows = []
    worst_cvd, worst_normal = math.inf, math.inf
    for i, j in pairlist:
        a, b = palette[i], palette[j]
        p, d, t = (delta_e(a, b, k) for k in ("protan", "deutan", "tritan"))
        n = delta_e(a, b)
        cvd = min(p, d)
        worst_cvd, worst_normal = min(worst_cvd, cvd), min(worst_normal, n)
        if n < NORMAL_FLOOR:
            st = "FAIL normal-vision"
            failed += 1
        elif cvd >= CVD_TARGET:
            st = "PASS"
        elif cvd >= CVD_FLOOR:
            st = "WARN — needs secondary encoding"
        else:
            st = "FAIL cvd"
            failed += 1
        pair_rows.append(
            {
                "pair": f"{i + 1}-{j + 1}",
                "protan": round(p, 1),
                "deutan": round(d, 1),
                "tritan": round(t, 1),
                "normal": round(n, 1),
                "status": st,
            }
        )

    return {
        "mode": mode,
        "surface": surface,
        "pairs": pairs,
        "slots": rows,
        "pairs_detail": pair_rows,
        "worst_cvd": round(worst_cvd, 1) if pair_rows else None,
        "worst_normal": round(worst_normal, 1) if pair_rows else None,
        "failed": failed,
    }


def report(result: dict) -> None:
    print(f"mode={result['mode']}  surface={result['surface']}  pairs={result['pairs']}")
    print(f"{'slot':>4} {'hex':>9} {'L':>6} {'C':>6} {'contr':>6}  {'band':<14} {'chroma':<10} status")
    for r in result["slots"]:
        print(
            f"{r['slot']:>4} {r['hex']:>9} {r['L']:>6} {r['C']:>6} {r['contrast']:>6}  "
            f"{r['band']:<14} {r['chroma']:<10} {r['status']} {r['relief']}"
        )
    if result["pairs_detail"]:
        print(f"\n{'pair':>6} {'protan':>7} {'deutan':>7} {'tritan':>7} {'normal':>7}  status")
        for r in result["pairs_detail"]:
            print(
                f"{r['pair']:>6} {r['protan']:>7} {r['deutan']:>7} {r['tritan']:>7} {r['normal']:>7}  {r['status']}"
            )
        print(f"\nworst CVD dE {result['worst_cvd']} (target >={CVD_TARGET})   "
              f"worst normal dE {result['worst_normal']} (floor >={NORMAL_FLOOR})")
    print(f"\n{result['failed']} failure(s)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("palette", help="comma-separated hex values")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface")
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    args = ap.parse_args(argv)
    pal = [p.strip() for p in args.palette.split(",") if p.strip()]
    res = validate(pal, mode=args.mode, surface=args.surface, pairs=args.pairs)
    report(res)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
