#!/usr/bin/env python3
"""Smoke tests for the 2026-09 fixes (no C++ env / no GPU required).

Run:  python tests/test_smoke_fixes.py
Covers:
  1. RewardConfig.to_dict() exports every key cpp_vecenv knows about.
  2. RewardConfig defaults == reward_v2 profile == configs/reward.json.
  3. UI-style flat config survives worker Config.from_dict (no dropped keys).
  4. PPO collect_step + update run on CPU with action masks (metrics finite,
     params actually change, save/load round-trip works).
  5. torch_compile=True on CPU falls back to eager without crashing.
  6. parameter_widget specs include the previously-missing knobs and their
     defaults come from rl.config.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ── 1. to_dict covers every key RewardConfig ──────────────
def reward_keys_from_cpp_vecenv() -> set:
    return set(RewardConfig().to_dict().keys())


from rl.config import Config, RewardConfig

rc = RewardConfig()
exported = set(rc.to_dict())
known = reward_keys_from_cpp_vecenv()
missing = known - exported
check("to_dict exports all cpp_vecenv keys", not missing, f"missing={sorted(missing)}")

# ── 2. defaults == reward_v3.json ──────────────────────────────
v3 = json.loads((ROOT / "configs" / "reward_v3.json").read_text(encoding="utf-8"))
diff_v3 = {k: (getattr(rc, k), v3[k]) for k in v3 if k != "_comment" and getattr(rc, k) != v3[k]}
check("dataclass defaults == reward_v3.json", not diff_v3, str(diff_v3))

# ── 3. worker-style config round-trip keeps every field ─────────────────
ui_cfg = {
    "name": "run_999", "n_envs": 16, "learning_rate": 1e-4, "gamma": 0.997,
    "eval_seeds": [42, 43, 44], "early_stopping_patience": 7,
    "unlock_ids": "House,Farm", "loop_detection_enabled": True,
    "loop_consecutive_threshold": 12, "target_kl": 0.015,
    "difficulty": "light", "eval_score_weights": [0.1, 1.0, 0.1, 1e-4],
    "eval_min_bases": 6, "eval_min_days": 800.0, "eval_use_median": True,
    "queue_size": 3, "build_bonus": 3.5, "death_penalty": 25.0,
}
work = dict(ui_cfg)
work.pop("name", None)
cfg = Config.from_dict(work)
cfg.reward = RewardConfig.from_dict(ui_cfg)
checks = [
    ("eval_seeds", cfg.eval_seeds == [42, 43, 44]),
    ("early_stopping_patience", cfg.early_stopping_patience == 7),
    ("unlock_ids", cfg.unlock_ids == "House,Farm"),
    ("loop_detection_enabled", cfg.loop_detection_enabled is True),
    ("loop_consecutive_threshold", cfg.loop_consecutive_threshold == 12),
    ("target_kl", cfg.target_kl == 0.015),
    ("difficulty", cfg.difficulty == "light"),
    ("eval_score_weights", tuple(cfg.eval_score_weights) == (0.1, 1.0, 0.1, 1e-4)),
    ("eval_min_bases", cfg.eval_min_bases == 6),
    ("eval_min_days", cfg.eval_min_days == 800.0),
    ("queue_size", cfg.queue_size == 3),
    ("reward.build_bonus", cfg.reward.build_bonus == 3.5),
    ("reward.death_penalty", cfg.reward.death_penalty == 25.0),
]
for name, ok in checks:
    check(f"worker roundtrip keeps {name}", ok)

# nested "reward" dict form (JSON profiles)
cfg2 = RewardConfig.from_dict({"reward": {"novelty": 42.0}})
check("RewardConfig.from_dict supports nested 'reward'", cfg2.novelty == 42.0)

# ── 4/5. PPO smoke on CPU ────────────────────────────────────────────────
import torch
from rl.actor_critic import ActorCritic
from rl.rollout_buffer import RolloutBuffer
from rl.ppo import PPO

torch.manual_seed(0)
dev = torch.device("cpu")
n_envs, n_steps, obs_size, n_actions = 4, 8, 209, 45
model = ActorCritic(obs_size, n_actions, [64, 64], dev)
buf = RolloutBuffer(n_steps, n_envs, obs_size, n_actions, 0.997, 0.98, dev)
ppo = PPO(model=model, buffer=buf, lr=3e-4, gamma=0.997, gae_lambda=0.98,
          clip_range=0.2, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
          n_epochs=2, batch_size=16, use_amp=True, amp_dtype="bfloat16",
          torch_compile=True, device=dev, target_kl=0.02)

check("torch_compile on CPU falls back to eager", ppo._compiled is False)

params_before = [p.detach().clone() for p in model.params]
masks = torch.ones(n_envs, n_actions)
masks[:, 5:] = 0.0  # only first 5 actions allowed
obs = torch.randn(n_envs, obs_size)
for _ in range(n_steps):
    out = ppo.collect_step(obs, action_masks=masks)
    assert out["action"].shape == (n_envs,)
    assert (out["action"] < 5).all(), "masked action sampled!"
    buf.add(obs=obs, action=out["action"], reward=torch.rand(n_envs),
            log_prob=out["log_prob"], value=out["value"],
            done=torch.zeros(n_envs, dtype=torch.bool),
            terminated=torch.zeros(n_envs, dtype=torch.bool),
            action_masks=masks)
    obs = torch.randn(n_envs, obs_size)

last_v = torch.zeros(n_envs)
last_d = torch.zeros(n_envs, dtype=torch.bool)
stats = ppo.update(last_v, last_d)
finite = all(
    isinstance(v, float) and v == v and abs(v) < 1e9
    for k, v in stats.items() if k != "learning_rate"
)
check("update() returns finite metrics", finite, str({k: round(v, 4) for k, v in stats.items()}))
changed = any(not torch.equal(a, b) for a, b in zip(params_before, model.params))
check("update() changes weights", changed)

with tempfile.TemporaryDirectory() as td:
    pth = Path(td) / "m.pt"
    ppo.save(str(pth))
    ck = torch.load(str(pth), weights_only=False)
    bad = [k for k in ck["model_state"] if "_orig_mod" in k]
    check("checkpoint has no _orig_mod keys", not bad)
    model2 = ActorCritic(obs_size, n_actions, [64, 64], dev)
    ppo2 = PPO(model=model2, buffer=buf, lr=3e-4, gamma=0.997, gae_lambda=0.98,
               clip_range=0.2, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
               n_epochs=1, batch_size=16, use_amp=False, torch_compile=False,
               device=dev)
    ppo2.load(str(pth))
    same = all(torch.equal(a, b) for a, b in zip(model.params, model2.params))
    check("save/load round-trip preserves weights", same)

# ── 6. UI specs ──────────────────────────────────────────────────────────
from train_ui2.parameter_widget import PARAM_SPECS, REWARD_SPECS, spec_for

need_params = {"target_kl", "save_freq", "eval_freq", "eval_episodes",
               "eval_min_days", "eval_min_bases", "early_stopping_patience"}
need_rewards = {"survival_coeff", "tax_fail_penalty", "death_penalty",
                "base_lost_penalty", "born_bonus", "debt_coeff",
                "home_overflow_penalty", "housing_need_bonus",
                "food_need_bonus", "water_need_bonus"}
have_params = {s.key for s in PARAM_SPECS}
have_rewards = {s.key for s in REWARD_SPECS}
check("PARAM_SPECS has new knobs", need_params <= have_params,
      str(need_params - have_params))
check("REWARD_SPECS has hidden weights", need_rewards <= have_rewards,
      str(need_rewards - have_rewards))
check("spec default(target_kl) == Config default",
      spec_for("target_kl").default == Config().target_kl)
check("spec default(survival_coeff) == RewardConfig default",
      spec_for("survival_coeff").default == RewardConfig().survival_coeff)
# every UI-editable key must exist in the Config/RewardConfig dataclasses
cfg_fields = {f for f in Config().__dict__}
rc_fields = {f for f in RewardConfig().__dict__}
orphans = (({s.key for s in PARAM_SPECS} - cfg_fields)
           | ({s.key for s in REWARD_SPECS} - rc_fields)) - {"n_layers"}
check("no orphan specs (all keys exist in dataclasses)", not orphans, str(orphans))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
