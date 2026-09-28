#!/usr/bin/env python3
"""Поиск лучшего плана застройки на 2 года (перебор сценариев поверх симулятора).

Симулятор — tests/cpp/build_plan_probe.cpp: он проигрывает сценарий «дата ▸
постройка» на настоящем движке игры (Game), сам держит жильё, торгует на рынке,
берёт и гасит кредит, платит налоги. Здесь — только перебор самих сценариев.

Запуск (из корня репозитория, после сборки /tmp/plan):
    python3 tests/cpp/search_plan.py --stage grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BIN = "/tmp/plan"
RE_TOTALS = re.compile(
    r"деньги=(-?\d+) склад=(-?\d+) долг=(-?\d+) КЭШ=(-?\d+) активы=(-?\d+) ИТОГО=(-?\d+)"
)
RE_HEAD = re.compile(r"население=(\d+) жильё=(\d+) построек=(\d+)")


@dataclass
class Plan:
    """Сценарий: список (дата, постройка, количество)."""

    items: list[tuple[str, str, int]] = field(default_factory=list)
    pop_cap: int = 400
    credit: int = 460000

    def text(self) -> str:
        return "\n".join(f"{d} {b} {n}" for d, b, n in self.items) + "\n"

    def label(self) -> str:
        return "; ".join(f"{d[2:]}·{b}×{n}" for d, b, n in self.items)


@dataclass
class Outcome:
    cash: int
    assets: int
    total: int
    debt: int
    people: int
    bases: int
    blocked: str
    counts: dict[str, int]


def run(plan: Plan, seed: int = 42) -> Outcome:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(plan.text())
        path = fh.name
    out = subprocess.run(
        [BIN, "--seed", str(seed), "--plan", path, "--credit", str(plan.credit),
         "--pop-cap", str(plan.pop_cap)],
        capture_output=True, text=True, check=True,
    ).stdout
    Path(path).unlink(missing_ok=True)
    m = RE_TOTALS.search(out)
    h = RE_HEAD.search(out)
    if not m or not h:
        raise RuntimeError(f"не разобрать вывод:\n{out}")
    counts = dict(
        (k, int(v)) for k, v in re.findall(r"(\w+)×(\d+)", out.split("состав:")[1])
    )
    blocked = ""
    if "BLOCKED" in out:
        blocked = out.split("BLOCKED:")[1].strip().split("\n")[0][:40]
    return Outcome(
        cash=int(m.group(4)), assets=int(m.group(5)), total=int(m.group(6)),
        debt=int(m.group(3)), people=int(h.group(1)), bases=int(h.group(3)),
        blocked=blocked, counts=counts,
    )


def score(plan: Plan, seeds: tuple[int, ...]) -> tuple[float, list[Outcome]]:
    outs = [run(plan, s) for s in seeds]
    return sum(o.total for o in outs) / len(outs), outs


def grid(seeds: tuple[int, ...]) -> list[tuple[float, Plan, Outcome]]:
    """Основная сетка: роддома (когда/сколько) × нефть × прииски (когда/сколько)."""
    results = []
    pu_dates = ["1890-03-01"]
    pu2_dates = [None, "1890-09-01", "1890-11-01", "1891-01-01"]
    for pu1 in (0, 1, 2, 3):
        for pu2_date, pu2 in itertools.product(pu2_dates, (0, 1, 2, 3)):
            if (pu2_date is None) != (pu2 == 0):
                continue
            for oil in (0, 1):
                for m1_date in ("1891-04-25", "1891-05-12"):
                    for m1 in (2, 3, 4, 5, 6):
                        for m2 in (0, 1, 2):
                            items = []
                            if pu1:
                                items.append((pu_dates[0], "Puerperal", pu1))
                            if oil:
                                items.append(("1890-03-01", "Refinery", oil))
                            if pu2:
                                items.append((pu2_date, "Puerperal", pu2))
                            items.append((m1_date, "Goldmine", m1))
                            if m2:
                                items.append(("1891-06-20", "Goldmine", m2))
                            n_pu = pu1 + pu2
                            cap = 30 * (m1 + m2) + 3 * n_pu + 10 * oil + 30
                            plan = Plan(items=items, pop_cap=cap)
                            try:
                                s, outs = score(plan, seeds)
                            except Exception as exc:  # noqa: BLE001
                                print("ошибка", plan.label(), exc)
                                continue
                            results.append((s, plan, outs[0]))
    results.sort(key=lambda r: -r[0])
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="grid")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    res = grid(seeds)
    print(f"вариантов: {len(res)}   сиды: {seeds}")
    print(f"{'ИТОГО':>10} {'кэш':>9} {'активы':>9} {'людей':>6}  план")
    for s, plan, o in res[: args.top]:
        print(f"{s:>10.0f} {o.cash:>9} {o.assets:>9} {o.people:>6}  {plan.label()}"
              f"{'  ⛔' + o.blocked if o.blocked else ''}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"score": s, "plan": p.items, "pop_cap": p.pop_cap, "cash": o.cash,
              "assets": o.assets, "people": o.people} for s, p, o in res[:50]],
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
