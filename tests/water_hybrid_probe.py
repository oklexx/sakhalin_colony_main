"""Зонды «flat против hybrid — почему водоканал ищется хуже» (2026-09-21).

Запуск (из корня репозитория, нужен собранный colony_cpp; часть 2 — torch):

    python tests/water_hybrid_probe.py

Части:
  P1 — согласованность пары (flat, minimap) на границах эпизодов (гипотеза о
       смешении эпизодов проверяется «отпечатком» рельефа — каналы 0..6
       миникарты статичны внутри эпизода и меняются при reset);
  P2 — бит-в-бит равенство flat-потоков в flat и hybrid (если да — режимы
       различаются только архитектурой сети и парой миникарты);
  P3 — «разбавление» flat-ветви в гибридной сети: вклад ветвей в joint-слой и
       чувствительность логита BUILD_WATERCHANNEL к water_dx/water_dy против
       обычного MLP.

Разбор результатов и замеры 2026-09-21 — docs/WATER_HYBRID_AB_2026_09.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

WATER_DX, WATER_DY = 246, 247  # индексы water_dx/dy в flat obs v0 (src/observation.cpp)


def land_sig(mm: np.ndarray) -> bytes:
    """Отпечаток рельефа (каналы 0..6): статичен внутри эпизода, новый при reset."""
    return np.ascontiguousarray(np.round(np.asarray(mm)[:7]).astype(np.int8)).tobytes()


# ── P1: пара (flat, minimap) на границах эпизодов ────────────────────────────

def p1_pair_on_done(n_steps: int = 9000) -> bool:
    from cpp_vecenv_minimap import CppVecEnvMinimap

    ok = True
    for mode in ("hybrid", "minimap"):
        env = CppVecEnvMinimap(n_envs=2, map_size=200, obs_mode=mode, seed=11)
        out = env.reset()
        mm0 = np.asarray(out[1] if mode == "hybrid" else out, np.float32)
        prev_sig = [land_sig(mm0[i]) for i in range(env.num_envs)]
        n_done = 0
        for t in range(n_steps):
            obs, _r, dones, infos = env.step(np.zeros(env.num_envs, np.int64))
            mm_now = np.asarray(env.minimap_obs(), np.float32)
            mm_pair = np.asarray(obs[1] if mode == "hybrid" else obs, np.float32)
            for i in range(env.num_envs):
                if not dones[i]:
                    assert land_sig(mm_pair[i]) == land_sig(mm_now[i])
                    prev_sig[i] = land_sig(mm_pair[i])
                    continue
                n_done += 1
                same = land_sig(mm_pair[i]) == land_sig(mm_now[i])
                is_old = land_sig(mm_pair[i]) == prev_sig[i]
                term = np.asarray(infos[i]["terminal_observation"], np.float32)
                flat_diff = (float(np.abs(np.asarray(obs[0][i]) - term).max())
                             if mode == "hybrid" else float("nan"))
                print(f"[P1/{mode}] t={t} done: пара == текущему состоянию: {same}  "
                      f"пара == старому эпизоду: {is_old}  "
                      f"max|flat−terminal|∞={flat_diff:.3f}")
                ok = ok and same
                prev_sig[i] = land_sig(mm_now[i])
            if n_done >= 2:
                break
        env.close()
        if n_done == 0:
            print(f"[P1/{mode}] done не случился за {n_steps} шагов — зонд пассивен")
    print(f"[P1] {'OK' if ok else 'ПАРА СМЕШИВАЕТ СОСТОЯНИЯ'}")
    return ok


# ── P2: flat-потоки режимов идентичны ───────────────────────────────────────

def p2_flat_parity(n_steps: int = 20) -> bool:
    from cpp_vecenv import CppVecEnv
    from cpp_vecenv_minimap import CppVecEnvMinimap

    flat_env = CppVecEnv(n_envs=2, map_size=200, seed=42)
    hyb_env = CppVecEnvMinimap(n_envs=2, map_size=200, seed=42, obs_mode="hybrid")
    f0 = np.asarray(flat_env.reset(), np.float32)
    h0 = np.asarray(hyb_env.reset()[0], np.float32)
    worst = float(np.abs(f0 - h0).max())
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        a = rng.integers(0, 15, size=2)
        fo, fr, _fd, _fi = flat_env.step(a)
        ho, hr, _hd, _hi = hyb_env.step(a)
        worst = max(worst,
                    float(np.abs(np.asarray(fo, np.float32)
                                 - np.asarray(ho[0], np.float32)).max()),
                    float(np.abs(np.asarray(fr) - np.asarray(hr)).max()))
    flat_env.close()
    hyb_env.close()
    print(f"[P2] max|Δflat|∞ за {n_steps} шагов = {worst:.6f} → "
          f"{'потоки идентичны' if worst == 0.0 else 'ПОТОКИ РАСХОДЯТСЯ'}")
    return worst == 0.0


# ── P3: разбавление flat-ветви в гибридной сети ──────────────────────────────

def p3_branch_dilution(n_samples: int = 256) -> bool:
    import torch

    from rl.actor_critic import ActorCritic
    from rl.actor_critic_hybrid import ActorCriticHybrid
    from rl.config import Config, load_default_reward_config
    from rl.env_manager import EnvManager

    device = torch.device("cpu")
    cfg = Config()
    cfg.obs_mode = "hybrid"
    cfg.n_envs = 4
    cfg.use_amp = False
    cfg.reward = load_default_reward_config()
    em = EnvManager(cfg, device)
    names = em.action_names
    wc = names.index("BUILD_WATERCHANNEL")
    rng = np.random.default_rng(0)
    em.reset()
    flats, minis, masks = [], [], []
    for _ in range(n_samples // cfg.n_envs):
        m = np.asarray(em.vec_env.action_masks, np.float32)
        acts = np.empty(cfg.n_envs, np.int64)
        for i in range(cfg.n_envs):
            legal = np.flatnonzero(m[i] > 0)
            acts[i] = legal[rng.integers(len(legal))]
        obs, _r, _d, _i = em.vec_env.step(acts)
        flats.append(torch.as_tensor(np.asarray(obs[0]), dtype=torch.float32))
        minis.append(torch.as_tensor(np.asarray(obs[1]), dtype=torch.float32))
        masks.append(torch.as_tensor(m.copy()))
    em.close()
    flat = torch.cat(flats)
    mini = torch.cat(minis)
    mask = torch.cat(masks)

    def sens_hybrid(seed: int) -> tuple[float, float, float]:
        torch.manual_seed(seed)
        model = ActorCriticHybrid(obs_size=flat.shape[1], grid_size=32,
                                  n_actions=len(names), hidden_sizes=[256, 256],
                                  device=device)
        h_flat = model.flat_trunk(flat)
        # ребаланс 2026-09-22: CNN сплющивается через пул 2×2 и cnn_head в
        # 256-мерную ветку — та же точка, что учится; z_flat/z_cnn сравнимы
        h_cnn = model.cnn_head(model.cnn_pool(model.cnn(mini)).flatten(1))
        lin = model.joint[0]
        z_flat = torch.cat([h_flat, torch.zeros_like(h_cnn)], 1)
        z_cnn = torch.cat([torch.zeros_like(h_flat), h_cnn], 1)
        w = lin.weight
        e_flat = float(torch.nn.functional.linear(z_flat, w, None)
                       .detach().pow(2).mean().sqrt())
        e_cnn = float(torch.nn.functional.linear(z_cnn, w, None)
                      .detach().pow(2).mean().sqrt())
        fx = flat.clone().requires_grad_(True)
        mx = mini.clone().requires_grad_(True)
        lg, _ = model.forward(fx, mx, action_masks=mask)
        lg[:, wc].sum().backward()
        s_water = float(fx.grad[:, [WATER_DX, WATER_DY]].abs().sum())
        return e_flat ** 2 / (e_flat ** 2 + e_cnn ** 2), s_water, float(mx.grad.abs().sum())

    def sens_flat(seed: int) -> float:
        torch.manual_seed(seed)
        model = ActorCritic(flat.shape[1], len(names), [256, 256], device)
        fx = flat.clone().requires_grad_(True)
        lg, _ = model.forward(fx, action_masks=mask)
        lg[:, wc].sum().backward()
        return float(fx.grad[:, [WATER_DX, WATER_DY]].abs().sum())

    hs = [sens_hybrid(100 + k) for k in range(5)]
    fs = [sens_flat(100 + k) for k in range(5)]
    share = np.mean([h[0] for h in hs])
    h_water = np.mean([h[1] for h in hs])
    f_water = np.mean(fs)
    print(f"[P3] доля flat-энергии в joint = {share * 100:.2f}%  "
          f"(до ребаланса ≈ 256/65792 = 0.39%, после ≈ 256/512 = 50%)")
    print(f"[P3] Σ|d logit_WC / d water_dx,dy|: hybrid={h_water:.2f}, "
          f"flat={f_water:.2f} → flat/hybrid = {f_water / h_water:.1f}×")
    print(f"[P3] Σ|d logit_WC / d minimap| (hybrid) = {np.mean([h[2] for h in hs]):.0f}")
    return True


def main() -> None:
    try:
        import colony_cpp  # noqa: F401
    except ImportError:
        print("colony_cpp не собран — зонды P1/P2 недоступны (make build)")
        sys.exit(2)
    p1_pair_on_done()
    p2_flat_parity()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch не установлен — P3 пропущен")
        return
    p3_branch_dilution()


if __name__ == "__main__":
    main()
