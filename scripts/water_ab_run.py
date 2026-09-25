"""Мини-A/B обучение: flat против hybrid (+hybridfix) — «ищется ли водоканал».

Считает те же метрики, что видит пользователь в мониторинге (доля действий), плюс
их разложение:
  wc_open_rate  = P(BUILD_WATERCHANNEL размаскирован)   — «нашёл воду» (навигация);
  press|open    = P(action=WC | размаскирован)          — «строит, когда можно»;
  wc_share      = доля WC среди всех действий           — метрика мониторинга;
  dirs_share    = доля ROAD_E/W/S/N + BUILD_ROAD        — «дорожная» активность.

Обучающий стек — реальный (EnvManager + PPO + rollout-буфер репозитория),
отличие от train.py только в уменьшенных гиперпараметрах (CPU) и логировании.

Запуск: python scripts/water_ab_run.py --mode flat|hybrid|hybridfix
        [--preset ''|stage1] [--seed 42] [--steps 81920] [--out DIR]
Сводка: python scripts/water_ab_analyze.py --results DIR
Разбор: docs/WATER_HYBRID_AB_2026_09.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/user/sakhalin_colony_main")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from rl.config import Config, load_default_reward_config  # noqa: E402
from rl.env_manager import EnvManager  # noqa: E402

N_ACTIONS = 49


def make_fixed_env_class():
    """Гибрид с исправленным порядком чтения миникарты (пара из одного состояния).

    Текущий CppVecEnvMinimap.step_wait() читает миникарту ДО super().step_wait(),
    но C++ авто-reset уже переписал result.obs пост-reset наблюдением: на done
    шаге пара = (flat нового эпизода, миникарта старого). Здесь миникарта
    читается ПОСЛЕ — как только что сделала это среда.
    """
    from cpp_vecenv import CppVecEnv
    from cpp_vecenv_minimap import CppVecEnvMinimap

    class FixedEnv(CppVecEnvMinimap):
        def step_wait(self):
            obs, rewards, dones, infos = CppVecEnv.step_wait(self)
            minimap_obs = np.ascontiguousarray(
                self.cpp_vec.minimap_batch(), dtype=np.float32
            )
            if self.obs_mode == "minimap":
                return minimap_obs, rewards, dones, infos
            if self.obs_mode == "hybrid":
                return (obs, minimap_obs), rewards, dones, infos
            return obs, rewards, dones, infos

    return FixedEnv


def build_cfg(mode: str, preset: str, seed: int, steps: int, ent_coef: float = 0.015) -> Config:
    cfg = Config()
    if preset == "stage1":
        from rl.curriculum import apply_stage1_preset
        apply_stage1_preset(cfg)
    cfg.obs_mode = "hybrid" if mode.startswith("hybrid") else "flat"
    cfg.seed = seed
    cfg.n_envs = 4
    cfg.n_steps = 512
    cfg.batch_size = 512
    cfg.n_epochs = 4
    cfg.ent_coef = ent_coef
    cfg.total_timesteps = steps
    cfg.use_amp = False
    cfg.reward = load_default_reward_config()
    return cfg


def bootstrap_value(em: EnvManager, obs, last_masks: torch.Tensor) -> torch.Tensor:
    model = em.ppo.model
    if em.obs_mode == "hybrid":
        flat, minimap = obs
        return model.get_value(flat, minimap, action_masks=last_masks)
    return model.get_value(obs, action_masks=last_masks)


def run(mode: str, preset: str, seed: int, steps: int, out_dir: Path,
        ent_coef: float = 0.015) -> None:
    tag = f"{mode}_{preset or 'stage0'}_s{seed}"
    if ent_coef != 0.015:
        tag += f"_ent{round(ent_coef * 1000)}"
    out_path = out_dir / f"{tag}.jsonl"
    if mode == "hybridfix":
        import cpp_vecenv_minimap as mm_mod
        mm_mod.CppVecEnvMinimap = make_fixed_env_class()

    cfg = build_cfg(mode, preset, seed, steps, ent_coef)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    t0 = time.perf_counter()

    em = EnvManager(cfg, device)
    names = em.action_names
    wc = names.index("BUILD_WATERCHANNEL")
    dirs = [names.index(n) for n in ("ROAD_E", "ROAD_W", "ROAD_S", "ROAD_N")]
    roads = names.index("BUILD_ROAD")
    print(f"[{tag}] actions={len(names)} wc_idx={wc} obs_mode={em.obs_mode}", flush=True)

    obs = em.reset()
    n_envs = em.n_envs
    total = 0
    roll = 0
    ep_ends = 0
    first_open_roll = first_press_roll = None
    hist = []
    with open(out_path, "w", encoding="utf-8") as fh:
        while total < steps:
            cnt = np.zeros(len(names), np.int64)
            wc_open = wc_press = dirs_cnt = roads_cnt = 0
            rew_sum = 0.0
            for _ in range(cfg.n_steps):
                masks = np.asarray(em.vec_env.action_masks, np.float32)
                wc_open += int((masks[:, wc] > 0).sum())
                obs, infos = em.collect_step(obs)
                pos = em.buffer.pos - 1
                acts = (em.buffer.actions[pos * n_envs:(pos + 1) * n_envs]
                        .cpu().numpy())
                for a in acts:
                    cnt[int(a)] += 1
                wc_press += int((acts == wc).sum())
                dirs_cnt += int(np.isin(acts, dirs).sum())
                roads_cnt += int((acts == roads).sum())
                rew_sum += float(em.buffer.rewards[pos * n_envs:(pos + 1) * n_envs].sum())
                for info in infos:
                    if info.get("episode") is not None:
                        ep_ends += 1
            total += cfg.n_steps * n_envs
            roll += 1

            # bootstrap + update — как в AsyncTrainer._collect_rollout/_update_ppo
            last_pos = em.buffer.pos - 1
            terminated = (em.buffer.terminated[last_pos * n_envs:(last_pos + 1) * n_envs]
                          .cpu().numpy())
            last_masks = torch.as_tensor(
                np.asarray(em.vec_env.action_masks, np.float32), device=device)
            with torch.no_grad():
                last_value = bootstrap_value(em, obs, last_masks)
            last_done = torch.as_tensor(terminated, device=device)
            stats = em.ppo.update(last_value=last_value, last_done=last_done)

            n = cfg.n_steps * n_envs
            wc_share = 100.0 * wc_press / n
            open_rate = 100.0 * wc_open / n
            press_open = 100.0 * wc_press / max(wc_open, 1)
            dirs_share = 100.0 * dirs_cnt / n
            if first_open_roll is None and wc_open > 0:
                first_open_roll = (roll, total)
            if first_press_roll is None and wc_press > 0:
                first_press_roll = (roll, total)
            row = dict(
                rollout=roll, steps=total,
                wc_share=round(wc_share, 3),
                wc_open_rate=round(open_rate, 3),
                press_open=round(press_open, 2),
                dirs_share=round(dirs_share, 3),
                roads_share=round(100.0 * roads_cnt / n, 3),
                mean_reward=round(rew_sum / n, 3),
                entropy=round(stats.get("entropy", 0.0), 4),
                ploss=round(stats.get("policy_loss", 0.0), 5),
                vloss=round(stats.get("value_loss", 0.0), 3),
                ep_ends=ep_ends,
            )
            hist.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{tag}] r{roll:03d} steps={total:6d} | wc_share={wc_share:5.2f}% "
                  f"wc_open={open_rate:5.2f}% press|open={press_open:5.1f}% "
                  f"dirs={dirs_share:4.1f}% road={row['roads_share']:4.1f}% | "
                  f"H={row['entropy']:.3f} ep_ends={ep_ends} "
                  f"({(total) / max(time.perf_counter() - t0, 1e-9):.0f} sps)", flush=True)

    summary = dict(
        tag=tag, mode=mode, preset=preset or "stage0", seed=seed, steps=total,
        first_open_roll=first_open_roll, first_press_roll=first_press_roll,
        mean_wc_share_last5=float(np.mean([r["wc_share"] for r in hist[-5:]])),
        mean_open_last5=float(np.mean([r["wc_open_rate"] for r in hist[-5:]])),
        mean_press_open_last5=float(np.mean([r["press_open"] for r in hist[-5:]])),
        mean_dirs_last5=float(np.mean([r["dirs_share"] for r in hist[-5:]])),
        wall_s=round(time.perf_counter() - t0, 1),
    )
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] SUMMARY {json.dumps(summary)}", flush=True)
    em.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["flat", "hybrid", "hybridfix"])
    ap.add_argument("--preset", default="", choices=["", "stage1"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=61440)
    ap.add_argument("--ent-coef", type=float, default=0.015,
                    help="энтропийный коэффициент (0.015 — база мини-A/B)")
    ap.add_argument("--out", default="results_water_ab")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run(args.mode, args.preset, args.seed, args.steps, out_dir, args.ent_coef)


if __name__ == "__main__":
    main()
