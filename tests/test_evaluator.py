import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    from cpp_env import CppColonyEnv  # noqa: F401
    ENV_OK = True
except Exception:
    ENV_OK = False


def _make_actor_critic_checkpoint(tmp_path: Path) -> Path:
    from rl.actor_critic import ActorCritic

    obs_size, n_actions, hidden = 203, 45, [64]
    m = ActorCritic(obs_size=obs_size, n_actions=n_actions,
                    hidden_sizes=hidden, device=torch.device("cpu"))
    ckpt_path = tmp_path / "fake_model.pt"
    torch.save({
        "model_state": m.state_dict(),
        "obs_size": obs_size,
        "n_actions": n_actions,
        "hidden_sizes": hidden,
    }, str(ckpt_path))
    return ckpt_path


def _make_legacy_checkpoint(tmp_path: Path) -> Path:
    import torch.nn as nn

    m = nn.Sequential(nn.Linear(4, 2))
    ckpt_path = tmp_path / "legacy.pt"
    torch.save({"model_state": m.state_dict()}, str(ckpt_path))
    return ckpt_path


def _fake_env_class():
    import numpy as np

    class _FakeNormalizer:
        def __init__(self):
            self.loaded = False
            self.update_enabled = True

        def load(self, path):
            self.loaded = True

        def set_update(self, enable):
            self.update_enabled = enable

    class FakeEnv:
        def __init__(self, **kw):
            self._step = 0
            self.normalizer = _FakeNormalizer()

        def reset(self, seed=None):
            self._step = 0
            return np.zeros(203, dtype=np.float32), {"days": 0}

        def step(self, action):
            self._step += 1
            obs = np.zeros(203, dtype=np.float32)
            info = {"days": self._step, "people": 10 + self._step, "bases": 3 + self._step}
            terminated = self._step >= 5
            return obs, 1.0, terminated, False, info

        def close(self):
            pass

    return FakeEnv


def test_load_policy_with_meta(tmp_path):
    if not ENV_OK:
        pytest.skip("env not available")
    from train_ui2.evaluator import _load_policy

    ckpt = _make_actor_critic_checkpoint(tmp_path)
    policy = _load_policy(ckpt, torch.device("cpu"))
    assert policy.obs_size == 203
    assert policy.n_actions == 45


def test_load_policy_probes_env_when_meta_missing(tmp_path):
    if not ENV_OK:
        pytest.skip("env not available")
    from train_ui2.evaluator import _load_policy

    from rl.actor_critic import ActorCritic
    m = ActorCritic(obs_size=203, n_actions=45, hidden_sizes=[256, 256],
                    device=torch.device("cpu"))
    ckpt_path = tmp_path / "nometa.pt"
    torch.save({"model_state": m.state_dict()}, str(ckpt_path))

    policy = _load_policy(ckpt_path, torch.device("cpu"))
    assert policy.obs_size == 203
    assert policy.n_actions == 45


def test_load_policy_rejects_incompatible(tmp_path):
    from train_ui2.evaluator import _load_policy

    ckpt = _make_legacy_checkpoint(tmp_path)
    with pytest.raises(ValueError):
        _load_policy(ckpt, torch.device("cpu"))


def test_run_eval_returns_stats(tmp_path, monkeypatch):
    if not ENV_OK:
        pytest.skip("env not available")
    import train_ui2.evaluator as ev

    ckpt = _make_actor_critic_checkpoint(tmp_path)
    fake_env = _fake_env_class()
    monkeypatch.setattr(ev, "CppColonyEnv", fake_env, raising=False)

    class FakePolicy:
        def __call__(self, obs):
            import torch
            return torch.zeros(1, 45), torch.zeros(1, 1)

    monkeypatch.setattr(ev, "_load_policy", lambda *a, **kw: FakePolicy())
    result = ev.run_eval(ckpt, episodes=2, max_days=20, seed=1, device="cpu")

    assert result["episodes"] == 2.0
    assert result["days"] >= 0
    assert result["people"] >= 0
    assert result["bases"] >= 0
    assert result["avg_return"] >= 0


@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_with_normalization(tmp_path, monkeypatch):
    """Test that run_eval loads normalization when provided."""
    import train_ui2.evaluator as ev
    import numpy as np

    ckpt = _make_actor_critic_checkpoint(tmp_path)

    import cpp_env as cpp_env_mod

    real_obs = int(cpp_env_mod.CppColonyEnv(
        map_size=100).observation_space.shape[0])
    import json
    norm_data = {
        "mean": [0.0] * real_obs,
        "var": [1.0] * real_obs,
        "count": 100,
        "obs_size": real_obs,
        "clip": 10.0,
    }
    norm_path = tmp_path / "norm.json"
    with open(norm_path, "w") as f:
        json.dump(norm_data, f)

    # Patch the real CppColonyEnv from cpp_env module (run_eval imports from there)
    from cpp_env import CppColonyEnv as RealCppColonyEnv
    import cpp_env

    instances = []
    original_init = RealCppColonyEnv.__init__
    def _capture_init(self, **kw):
        original_init(self, **kw)
        instances.append(self)
    monkeypatch.setattr(cpp_env, "CppColonyEnv", type(
        "PatchedCppColonyEnv", (RealCppColonyEnv,), {"__init__": _capture_init}
    ))

    class FakePolicy:
        def __call__(self, obs):
            return torch.zeros(1, 45), torch.zeros(1, 1)

    monkeypatch.setattr(ev, "_load_policy", lambda *a, **kw: FakePolicy())

    result = ev.run_eval(ckpt, episodes=1, max_days=10, seed=1, device="cpu",
                         normalization_path=str(norm_path))

    assert result["episodes"] == 1.0
    assert len(instances) == 1, "expected exactly one env instance to be created"
    env = instances[0]
    # The real Normalizer should have loaded the stats and disabled updates
    assert env.normalizer._update_enabled is False


def test_run_eval_missing_model(tmp_path):
    from train_ui2.evaluator import run_eval
    with pytest.raises(FileNotFoundError):
        run_eval(tmp_path / "nope.pt", episodes=1, max_days=1)


def _make_hybrid_checkpoint(tmp_path: Path, grid_size: int = 57,
                            obs_size: int | None = None) -> Path:
    """Hybrid policy checkpoint as saved by rl/ppo.py (grid_size in extras)."""
    from rl.actor_critic_hybrid import ActorCriticHybrid

    if obs_size is None:  # default: match the current default env layout
        import cpp_env as cpp_env_mod

        obs_size = int(cpp_env_mod.CppColonyEnv(
            map_size=100).observation_space.shape[0])
    m = ActorCriticHybrid(obs_size=obs_size, n_channels=8, grid_size=grid_size,
                          n_actions=45, hidden_sizes=[64], device="cpu")
    ckpt_path = tmp_path / "hybrid_model.pt"
    torch.save({
        "model_state": m.state_dict(),
        "obs_size": obs_size,
        "n_actions": 45,
        "hidden_sizes": [64],
        "n_channels": 8,
        "grid_size": grid_size,
    }, str(ckpt_path))
    return ckpt_path


@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_hybrid_pushes_policy_grid_into_env(tmp_path, monkeypatch):
    """Regression: a hybrid policy trained with minimap_radius != 14 crashed with
    "mat1 and mat2 shapes cannot be multiplied (1x3136 and 12544x256)" because the
    eval env kept the default radius 14 (grid 29) while the CNN expected grid 57.
    """
    import train_ui2.evaluator as ev
    import cpp_env as cpp_env_mod

    ckpt = _make_hybrid_checkpoint(tmp_path, grid_size=57)

    created = []
    RealEnv = cpp_env_mod.CppColonyEnv

    class SpyEnv(RealEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    monkeypatch.setattr(cpp_env_mod, "CppColonyEnv", SpyEnv, raising=False)

    result = ev.run_eval(ckpt, episodes=1, max_days=3, seed=1, device="cpu",
                         map_size=100, mode="hybrid", minimap_radius=28)

    assert result["episodes"] == 1.0
    assert len(created) == 1
    # radius derived from the policy's grid, not the default 14
    assert int(created[0].cpp_env.minimap_radius()) == 28


@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_flat_untouched_by_radius_fix(tmp_path, monkeypatch):
    """The flat path must not touch minimap radius (no minimap wrapper there)."""
    import train_ui2.evaluator as ev
    import cpp_env as cpp_env_mod

    real_obs = int(cpp_env_mod.CppColonyEnv(map_size=100).observation_space.shape[0])
    from rl.actor_critic import ActorCritic

    m = ActorCritic(obs_size=real_obs, n_actions=45, hidden_sizes=[64],
                    device=torch.device("cpu"))
    ckpt = tmp_path / "flat_model.pt"
    torch.save({"model_state": m.state_dict(), "obs_size": real_obs,
                "n_actions": 45, "hidden_sizes": [64]}, str(ckpt))

    created = []
    RealEnv = cpp_env_mod.CppColonyEnv

    class SpyEnv(RealEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    monkeypatch.setattr(cpp_env_mod, "CppColonyEnv", SpyEnv, raising=False)

    result = ev.run_eval(ckpt, episodes=1, max_days=3, seed=1, device="cpu",
                         map_size=100, mode="flat")

    assert result["episodes"] == 1.0
    assert int(created[0].cpp_env.minimap_radius()) == 14


@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_obs_mismatch_raises(tmp_path):
    """A v0-sized policy on the default v1 env must fail with 'obs mismatch'."""
    from rl.actor_critic import ActorCritic
    from train_ui2.evaluator import run_eval

    m = ActorCritic(obs_size=246, n_actions=45, hidden_sizes=[64],
                    device=torch.device("cpu"))
    ckpt = tmp_path / "v0_model.pt"
    torch.save({"model_state": m.state_dict(), "obs_size": 246,
                "n_actions": 45, "hidden_sizes": [64], "obs_version": 0},
               str(ckpt))
    with pytest.raises(RuntimeError, match="obs mismatch"):
        run_eval(ckpt, episodes=1, max_days=2, seed=1, device="cpu",
                 map_size=100)


@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_obs_version_0_runs_v0_policy(tmp_path):
    """The same v0 policy runs when the eval env is explicitly v0."""
    from rl.actor_critic import ActorCritic
    from train_ui2.evaluator import run_eval

    m = ActorCritic(obs_size=246, n_actions=45, hidden_sizes=[64],
                    device=torch.device("cpu"))
    ckpt = tmp_path / "v0_model.pt"
    torch.save({"model_state": m.state_dict(), "obs_size": 246,
                "n_actions": 45, "hidden_sizes": [64], "obs_version": 0},
               str(ckpt))
    result = run_eval(ckpt, episodes=1, max_days=2, seed=1, device="cpu",
                      map_size=100, obs_version=0)
    assert result["episodes"] == 1.0


@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_stored_meta_mismatch_raises(tmp_path):
    """Legacy meta (obs_version=0) + default v1 eval must fail loudly."""
    import json

    import cpp_env as cpp_env_mod
    from rl.actor_critic import ActorCritic
    from train_ui2.evaluator import run_eval

    real_obs = int(cpp_env_mod.CppColonyEnv(
        map_size=100).observation_space.shape[0])
    m = ActorCritic(obs_size=real_obs, n_actions=45, hidden_sizes=[64],
                    device=torch.device("cpu"))
    ckpt = tmp_path / "flat_model.pt"
    torch.save({"model_state": m.state_dict(), "obs_size": real_obs,
                "n_actions": 45, "hidden_sizes": [64]}, str(ckpt))
    (tmp_path / "meta.json").write_text(json.dumps({"obs_version": 0}),
                                        encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"obs v0.*246.*obs v1.*287"):
        run_eval(ckpt, episodes=1, max_days=2, seed=1, device="cpu",
                 map_size=100)
