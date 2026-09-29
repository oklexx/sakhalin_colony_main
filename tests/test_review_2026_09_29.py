"""Регрессии код-ревью обучающего контура от 2026-09-29.

Покрытые ошибки (подробности — CHANGELOG.md за 2026-09-29):

1. ``PPO._compute_loss_components``: оценка KL была ``|mean(old-new)|`` —
   положительные и отрицательные log-ratio гасили друг друга, и ранняя
   остановка по ``target_kl`` фактически не работала.
2. ``EnvManager._truncation_bootstrap_values`` получал маску шага s_t для
   терминального наблюдения s_T — чужой контекст для mask-проекций критика.
3. ``rl/bc_pretrain.py`` писал плоскую схему нормализации, а
   ``train.py --resume-model`` грузит файл в nested-загрузчик vec-среды —
   документированный путь «BC → PPO» был сломан.
4. ``final_model.pt`` сохранялся без meta.json — watch/eval восстанавливали
   сценарий по чужому (или легаси-) курикулуму.
5. ``early_stopping_patience`` тикал и до появления первого чемпиона —
   ранняя остановка убивала медленно стартующие прогоны.
6. ``curriculum_from_meta`` ставил бюджетный ``total_timesteps`` выше
   фактического ``steps`` при реконструкции точки расписания механик.
7. ``ActorCriticHybrid`` в flat-only вызове гарантированно умирал в matmul
   joint-слоя — теперь внятная ошибка.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.actor_critic import ActorCritic
from rl.actor_critic_hybrid import ActorCriticHybrid
from rl.async_trainer import AsyncTrainer
from rl.config import Config
from rl.curriculum import curriculum_from_meta
from rl.env_manager import EnvManager
from rl.ppo import PPO
from rl.rollout_buffer import RolloutBuffer

DEVICE = torch.device("cpu")


# ── 1. Оценщик KL ────────────────────────────────────────────────────────────

def _ppo_with_tiny_model() -> PPO:
    model = ActorCritic(obs_size=4, n_actions=3, hidden_sizes=[8], device=DEVICE)
    buffer = RolloutBuffer(n_steps=2, n_envs=2, obs_size=4, n_actions=3,
                           gamma=0.99, gae_lambda=0.95, device=DEVICE)
    return PPO(model=model, buffer=buffer, device=DEVICE, use_amp=False,
               n_epochs=1, batch_size=2)


def test_approx_kl_uses_nonnegative_estimator() -> None:
    """KL считается как mean((r-1) - log r), а не |mean(old-new)|.

    Подборка с log_ratio = [-0.5, +0.5]: старый эстиматор даёт ровно 0.0
    (полная компенсация), новый — ~0.128 (реальная divergence).
    """
    ppo = _ppo_with_tiny_model()
    obs = torch.randn(2, 4)
    actions = torch.tensor([0, 1])
    with torch.no_grad():
        logits, values = ppo.model(obs, None)
        new_lps = torch.distributions.Categorical(logits=logits).log_prob(actions)
    # old = новая политика, сдвинутая на ±0.5 — old-style |mean(old-new)| = 0
    old_lps = new_lps + torch.tensor([0.5, -0.5])
    zeros = torch.zeros(2)

    _pl, _vl, _ent, approx_kl = ppo._compute_loss_components(
        obs, actions, old_lps, zeros, zeros, zeros)

    log_ratio = new_lps - old_lps
    expected = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
    assert approx_kl.item() == pytest.approx(expected.item(), abs=1e-6)
    assert approx_kl.item() > 0.1  # старый эстиматор дал бы здесь ровно 0.0


def test_approx_kl_is_zero_for_identical_policies() -> None:
    """old == new (первый минибатч первой эпохи) → KL ≈ 0, не отрицательный."""
    ppo = _ppo_with_tiny_model()
    obs = torch.randn(2, 4)
    actions = torch.tensor([0, 1])
    with torch.no_grad():
        logits, _ = ppo.model(obs, None)
        lps = torch.distributions.Categorical(logits=logits).log_prob(actions)
    zeros = torch.zeros(2)
    _pl, _vl, _ent, approx_kl = ppo._compute_loss_components(
        obs, actions, lps.clone(), zeros, zeros, zeros)
    assert 0.0 <= approx_kl.item() < 1e-5


# ── 2. Маска усечённого бутстрапа ────────────────────────────────────────────

class _RecordingModel:
    """get_value-заглушка, запоминающая полученную маску."""

    def __init__(self) -> None:
        self.seen_masks: list = []

    def get_value(self, x, action_masks=None):
        self.seen_masks.append(action_masks)
        return torch.full((x.shape[0],), 5.0)


def _bare_env_manager(n_envs: int = 3) -> EnvManager:
    em = EnvManager.__new__(EnvManager)
    em.n_envs = n_envs
    em.device = DEVICE
    em.cfg = types.SimpleNamespace(obs_mode="flat")
    return em


def test_truncation_bootstrap_uses_terminal_mask_when_available() -> None:
    """Маска s_T из info идёт в критик; без неё — action_masks=None.

    До фикса сюда подставлялась маска шага s_t (аргумент метода) — чужое
    состояние для critic_mask_proj.
    """
    em = _bare_env_manager()
    model = _RecordingModel()
    em.model = model
    term_mask = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    infos = [
        {"terminal_observation": np.zeros(4, dtype=np.float32),
         "terminal_action_mask": term_mask[0]},
        {"terminal_observation": np.zeros(4, dtype=np.float32)},  # маски нет
        {"terminal_observation": np.zeros(4, dtype=np.float32),
         "terminal_action_mask": term_mask[0]},
    ]
    out = em._truncation_bootstrap_values(infos, np.array([True, True, True]))

    assert out.tolist() == [5.0, 5.0, 5.0]
    assert len(model.seen_masks) == 2  # два вызова: с масками и без
    masked = [m for m in model.seen_masks if m is not None]
    plain = [m for m in model.seen_masks if m is None]
    assert len(masked) == 1 and len(plain) == 1
    # батч с масками собран ровно из двух terminal_action_mask (env 0 и 2)
    np.testing.assert_allclose(masked[0].cpu().numpy(),
                               np.stack([term_mask[0], term_mask[0]]))


def test_truncation_bootstrap_skips_missing_terminal_obs() -> None:
    """Среда без terminal_observation честно пропускается (бутстрап = 0)."""
    em = _bare_env_manager(n_envs=2)
    em.model = _RecordingModel()
    infos = [{"terminal_observation_missing": True},
             {"terminal_observation": np.zeros(4, dtype=np.float32)}]
    out = em._truncation_bootstrap_values(infos, np.array([True, True]))
    assert out.tolist() == [0.0, 5.0]
    # единственный вызов — без маски (terminal_action_mask не прислали)
    assert em.model.seen_masks == [None]


# ── 3. Схема нормализации BC ─────────────────────────────────────────────────

def test_bc_vec_normalization_payload_schema() -> None:
    """Файл BC грузится обоими читателями: vec-env (nested) и single (obs_size)."""
    from rl.bc_pretrain import vec_normalization_payload

    payload = vec_normalization_payload(
        {"mean": [1.0, 2.0], "var": [4.0, 9.0], "count": 7,
         "obs_size": 2, "clip": 5.0})
    # nested-схема ColonyVecEnvCpp::save_normalization
    assert payload["obs_rms"]["mean"] == [1.0, 2.0]
    assert payload["obs_rms"]["var"] == [4.0, 9.0]
    assert payload["obs_rms"]["count"] == 7.0
    assert payload["rew_rms"] == {"mean": [0.0], "var": [1.0], "count": 1.0}
    assert payload["norm_obs"] is True and payload["norm_reward"] is True
    assert payload["clip_obs"] == 5.0 and payload["clip_reward"] == 10.0
    # верхнеуровневый extra-ключ для PR 5-проверки одиночной среды
    assert payload["obs_size"] == 2


# ── 4/5. Полный фейковый прогон: meta финалки + ранняя остановка ─────────────

class _FakeEnvManager:
    """Минимальный EnvManager: 2 среды × 3 шага, эпизодов нет."""

    def __init__(self, n_steps: int = 3) -> None:
        self.n_envs = 2
        self.obs_size = 10
        self.n_actions = 5
        self.device = DEVICE
        self._step_count = 0
        self.cfg = type("FakeCfg", (), {"map_size": 280, "obs_mode": "flat"})()
        torch.manual_seed(0)
        self.model = ActorCritic(10, 5, [16], self.device)
        self.buffer = RolloutBuffer(
            n_steps=n_steps, n_envs=2, obs_size=10, n_actions=5,
            gamma=0.99, gae_lambda=0.95, device=self.device)
        self.ppo = PPO(model=self.model, buffer=self.buffer, n_epochs=1,
                       batch_size=4, use_amp=False, device=self.device)
        self.vec_env = type("FakeEnv", (), {})()
        self.vec_env.venv = type("FakeVenv", (), {
            "save_normalization": staticmethod(lambda path: None)})()

    @property
    def action_names(self):
        return [f"A{i}" for i in range(5)]

    def reset(self):
        return torch.randn(self.n_envs, self.obs_size)

    def collect_step(self, obs):
        self._step_count += 1
        with torch.no_grad():
            action, log_prob, value = self.model.get_action_and_value(obs)
        new_obs = torch.randn(self.n_envs, self.obs_size)
        rewards = torch.randn(self.n_envs)
        dones = torch.zeros(self.n_envs, dtype=torch.bool)
        self.buffer.add(obs=obs, action=action, reward=rewards,
                        log_prob=log_prob, value=value, done=dones)
        return new_obs, [{}, {}]

    def close(self):
        pass


def _train_cfg(tmp_path, **kw) -> Config:
    base = dict(n_envs=2, n_steps=3, total_timesteps=18, save_freq=0,
                eval_freq=0, use_amp=False, model_dir=str(tmp_path))
    base.update(kw)
    return Config(**base)


def test_final_model_meta_json_is_written(tmp_path) -> None:
    """Финалка получает meta.json с тем же curriculum-снимком, что и чекпойнты."""
    import json as _json

    cfg = _train_cfg(tmp_path)
    trainer = AsyncTrainer(cfg=cfg, env_manager=_FakeEnvManager())
    # run_eval в песочнице недоступен (нет colony_cpp): турнир по кандидатам
    # падает под защитой per-candidate try/except — это норма для этого теста.
    trainer.train(total_timesteps=cfg.total_timesteps)

    meta_path = tmp_path / "final_model.meta.json"
    assert meta_path.exists(), "final_model.meta.json не записан"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["total_timesteps"] == cfg.total_timesteps
    assert "enabled_mechanics" in meta
    assert "obs_version" in meta


def test_early_stop_patience_waits_for_first_baseline(tmp_path) -> None:
    """Без чемпиона patience не тикает: прогон доезжает до total_timesteps."""
    cfg = _train_cfg(tmp_path, eval_freq=6, early_stopping_patience=2,
                     total_timesteps=18)
    trainer = AsyncTrainer(cfg=cfg, env_manager=_FakeEnvManager())
    eval_calls = []

    def _fake_eval(total_done: int) -> dict:
        eval_calls.append(total_done)
        # thresholds не пройдены → best_score остаётся None навсегда
        return {"days": 1.0, "bases": 0.0, "people": 0.0,
                "water": 0.0, "water_ready": 0.0, "avg_return": -5.0,
                "score": 0.0, "saved": False}

    trainer._eval = _fake_eval  # type: ignore[method-assign]
    trainer.train(total_timesteps=cfg.total_timesteps)

    assert len(eval_calls) == 3  # eval на каждом из 3 роллаутов
    # До фикса: после 2-го eval patience исчерпывался и прогон стопался
    # раньше total_timesteps без всякой базовой линии.
    assert trainer.metrics.stop_reason == ""
    assert trainer.metrics.total_timesteps == cfg.total_timesteps


def test_early_stop_fires_after_baseline_stalls(tmp_path) -> None:
    """Классика: чемпион есть и не растёт → прогон останавливается по patience."""
    cfg = _train_cfg(tmp_path, eval_freq=6, early_stopping_patience=2,
                     total_timesteps=60)
    trainer = AsyncTrainer(cfg=cfg, env_manager=_FakeEnvManager())

    def _fake_eval(total_done: int) -> dict:
        if trainer.best_score is None:
            trainer.best_score = 10.0  # первый чемпион появился
        return {"days": 800.0, "bases": 6.0, "people": 100.0,
                "water": 0.0, "water_ready": 0.0, "avg_return": 5.0,
                "score": 10.0, "saved": False}

    trainer._eval = _fake_eval  # type: ignore[method-assign]
    trainer.train(total_timesteps=cfg.total_timesteps)

    assert trainer.metrics.stop_reason == "early_stop"
    assert trainer.metrics.total_timesteps < cfg.total_timesteps


# ── 6. Точка реконструкции расписания механик ────────────────────────────────

def test_mechanics_step_prefers_actual_steps_over_budget() -> None:
    """run meta воркера: «steps» (факт) важнее «config.total_timesteps» (бюджет)."""
    meta = {"steps": 100, "config": {"total_timesteps": 1000}}
    assert curriculum_from_meta(meta)["mechanics_step"] == 100


# ── 7. Гибрид: flat-only вызов — внятная ошибка ─────────────────────────────

def test_hybrid_flat_only_fails_loudly() -> None:
    model = ActorCriticHybrid(obs_size=8, n_channels=8, grid_size=4,
                              n_actions=3, hidden_sizes=[8, 8], device=DEVICE)
    with pytest.raises(ValueError, match="BOTH branches"):
        model(torch.randn(2, 8))


def test_hybrid_pair_still_works() -> None:
    model = ActorCriticHybrid(obs_size=8, n_channels=8, grid_size=4,
                              n_actions=3, hidden_sizes=[8, 8], device=DEVICE)
    logits, values = model(torch.randn(2, 8), torch.randn(2, 8, 4, 4))
    assert logits.shape == (2, 3) and values.shape == (2, 1)
