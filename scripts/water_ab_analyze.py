"""Сводка мини-A/B: сравнение flat / hybrid / hybridfix по results/*.jsonl.

Запуск: python scripts/water_ab_analyze.py [--results DIR]
(результаты пишет scripts/water_ab_run.py; разбор — docs/WATER_HYBRID_AB_2026_09.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RES = Path("results_water_ab")


def load(tag: str):
    rows = [json.loads(line) for line in (RES / f"{tag}.jsonl").read_text().splitlines()]
    summ = json.loads((RES / f"{tag}.summary.json").read_text())
    return rows, summ


def series(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def smooth(a, w=5):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


def main() -> None:
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_water_ab",
                    help="каталог с *.jsonl / *.summary.json от water_ab_run.py")
    args = ap.parse_args()
    RES = Path(args.results)
    tags = sorted(p.name.removesuffix(".summary.json") for p in RES.glob("*.summary.json"))
    print(f"{'tag':24s} | {'steps':>6s} | {'wc_share_last5':>14s} | "
          f"{'open_last5':>10s} | {'press|open_last5':>15s} | {'dirs_last5':>9s} | "
          f"{'1st_press':>9s} | {'H_last5':>6s}")
    for t in tags:
        rows, s = load(t)
        H = float(np.mean([r["entropy"] for r in rows[-5:]]))
        fp = s.get("first_press_roll")
        fp_s = fp[1] if fp else "-"
        print(f"{t:24s} | {s['steps']:6d} | {s['mean_wc_share_last5']:14.3f} | "
              f"{s['mean_open_last5']:10.3f} | {s['mean_press_open_last5']:15.1f} | "
              f"{s['mean_dirs_last5']:9.1f} | {str(fp_s):>9s} | {H:6.2f}")

    print("\n— траектории (сглаживание по 5 роллаутов), wc_share% — метрика «Водоканал» в мониторинге:")
    for t in tags:
        rows, _ = load(t)
        sc = smooth(series(rows, "wc_share"))
        op = smooth(series(rows, "wc_open_rate"))
        po = smooth(series(rows, "press_open"))
        ent = smooth(series(rows, "entropy"))
        idx = [0, len(sc) // 4, len(sc) // 2, 3 * len(sc) // 4, len(sc) - 1]
        print(f"\n{t}")
        print("  wc_share: " + " ".join(f"{sc[i]:.2f}" for i in idx))
        print("  wc_open : " + " ".join(f"{op[i]:.2f}" for i in idx))
        print("  press|op: " + " ".join(f"{po[i]:.0f}" for i in idx))
        print("  entropy : " + " ".join(f"{ent[i]:.2f}" for i in idx))

    print("\n— парные отношения (hybrid/flat, одинаковый seed):")

    def ratio(hv: float, fv: float, xv: float) -> str:
        return f"hybrid/flat={hv / max(fv, 1e-9):.2f}, fix/flat={xv / max(fv, 1e-9):.2f}"

    for preset in ("stage0", "stage1"):
        try:
            _, f_s = load(f"flat_{preset}_s42")
            _, h_s = load(f"hybrid_{preset}_s42")
            _, x_s = load(f"hybridfix_{preset}_s42")
        except FileNotFoundError:
            continue
        for key, label in (("mean_wc_share_last5", "wc_share"),
                           ("mean_open_last5", "wc_open"),
                           ("mean_press_open_last5", "press|op")):
            print(f"{preset}: {label:9s} "
                  f"{ratio(h_s[key], f_s[key], x_s[key])}")


if __name__ == "__main__":
    main()
