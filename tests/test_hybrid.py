import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest


@pytest.fixture
def hybrid_model():
    from rl.actor_critic_hybrid import ActorCriticHybrid
    return ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[256, 256], device="cpu",
    )


def test_hybrid_forward_shapes(hybrid_model):
    flat = torch.randn(4, 289)
    mm = torch.randn(4, 8, 32, 32)
    logits, values = hybrid_model(flat, mm)
    assert logits.shape == (4, 45)
    assert values.shape == (4, 1)


def test_hybrid_act(hybrid_model):
    flat = torch.randn(289)
    mm = torch.randn(8, 32, 32)
    action, log_prob, value = hybrid_model.act(flat.unsqueeze(0), mm.unsqueeze(0))
    assert action.shape == (1,)
    assert 0 <= int(action.item()) < 45
    assert log_prob.shape == (1,)


def test_hybrid_act_deterministic(hybrid_model):
    flat = torch.randn(289)
    mm = torch.randn(8, 32, 32)
    a1, _, _ = hybrid_model.act(flat.unsqueeze(0), mm.unsqueeze(0), deterministic=True)
    a2, _, _ = hybrid_model.act(flat.unsqueeze(0), mm.unsqueeze(0), deterministic=True)
    assert a1.item() == a2.item()


def test_hybrid_state_dict_roundtrip(hybrid_model):
    sd = hybrid_model.state_dict()
    from rl.actor_critic_hybrid import ActorCriticHybrid
    m2 = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[256, 256], device="cpu",
    )
    m2.load_state_dict(sd)
    flat = torch.randn(289)
    mm = torch.randn(8, 32, 32)
    with torch.no_grad():
        a1, _, _ = hybrid_model.act(flat.unsqueeze(0), mm.unsqueeze(0), deterministic=True)
        a2, _, _ = m2.act(flat.unsqueeze(0), mm.unsqueeze(0), deterministic=True)
    assert a1.item() == a2.item()


def test_hybrid_get_value(hybrid_model):
    flat = torch.randn(2, 289)
    mm = torch.randn(2, 8, 32, 32)
    values = hybrid_model.get_value(flat, mm)
    assert values.shape == (2,)


def test_hybrid_get_action_and_value(hybrid_model):
    flat = torch.randn(3, 289)
    mm = torch.randn(3, 8, 32, 32)
    action, log_prob, value = hybrid_model.get_action_and_value(flat, mm)
    assert action.shape == (3,)
    assert log_prob.shape == (3,)
    assert value.shape == (3,)


def test_hybrid_params_property(hybrid_model):
    params = hybrid_model.params
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_hybrid_different_grid_size(hybrid_model):
    from rl.actor_critic_hybrid import ActorCriticHybrid
    m16 = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=33,
        n_actions=45, hidden_sizes=[128, 128], device="cpu",
    )
    flat = torch.randn(2, 289)
    mm = torch.randn(2, 8, 33, 33)
    logits, values = m16(flat, mm)
    assert logits.shape == (2, 45)
    assert values.shape == (2, 1)


def test_hybrid_gradient_flow(hybrid_model):
    hybrid_model.train()
    flat = torch.randn(4, 289, requires_grad=True)
    mm = torch.randn(4, 8, 32, 32, requires_grad=True)
    logits, values = hybrid_model(flat, mm)
    loss = logits.mean() + values.mean()
    loss.backward()
    assert flat.grad is not None
    assert mm.grad is not None
    assert hybrid_model.flat_proj.weight.grad is not None
    assert hybrid_model.cnn[0].weight.grad is not None
    assert hybrid_model.actor.weight.grad is not None
    assert hybrid_model.critic.weight.grad is not None


def test_hybrid_load_policy_detection(tmp_path):
    from rl.actor_critic_hybrid import ActorCriticHybrid
    from train_ui2.evaluator import _load_policy

    m = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[256, 256], device="cpu",
    )
    ckpt_path = tmp_path / "hybrid.pt"
    torch.save({
        "model_state": m.state_dict(),
        "obs_size": 289,
        "n_channels": 8,
        "grid_size": 32,
        "n_actions": 45,
        "hidden_sizes": [256, 256],
    }, str(ckpt_path))

    policy = _load_policy(ckpt_path, torch.device("cpu"))
    assert isinstance(policy, ActorCriticHybrid)
    assert policy.obs_size == 289
    assert policy.n_channels == 8
    assert policy.grid_size == 32
    assert policy.n_actions == 45


def test_hybrid_ppo_forward():
    import torch
    from rl.actor_critic_hybrid import ActorCriticHybrid
    from rl.rollout_buffer import _TensorRolloutBuffer
    from rl.ppo import PPO

    model = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[64, 64], device="cpu",
    )
    n_envs = 2
    n_steps = 4
    buf = _TensorRolloutBuffer(
        n_steps=n_steps, n_envs=n_envs,
        obs_shape=(8, 32, 32),
        n_actions=45,
        gamma=0.99, gae_lambda=0.95,
        device=torch.device("cpu"),
        flat_dim=289,
    )
    ppo = PPO(model=model, buffer=buf, device=torch.device("cpu"), use_amp=False)

    for t in range(n_steps):
        flat = torch.randn(n_envs, 289)
        mm = torch.randn(n_envs, 8, 32, 32)
        with torch.no_grad():
            a, lp, v = model.get_action_and_value(flat, mm)
        buf.add(
            obs=mm, action=a, reward=torch.ones(n_envs),
            log_prob=lp, value=v,
            done=torch.zeros(n_envs, dtype=torch.bool),
            terminated=torch.zeros(n_envs, dtype=torch.bool),
            flat=flat,
        )

    last_flat = torch.randn(n_envs, 289)
    last_mm = torch.randn(n_envs, 8, 32, 32)
    with torch.no_grad():
        last_value = model.get_value(last_flat, last_mm)
    last_done = torch.zeros(n_envs, dtype=torch.bool)

    stats = ppo.update(last_value=last_value, last_done=last_done)
    assert "policy_loss" in stats
    assert "value_loss" in stats


def test_hybrid_save_load_roundtrip(tmp_path):
    from rl.actor_critic_hybrid import ActorCriticHybrid
    from rl.rollout_buffer import _TensorRolloutBuffer
    from rl.ppo import PPO

    model = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[64, 64], device="cpu",
    )
    buf = _TensorRolloutBuffer(
        n_steps=4, n_envs=2,
        obs_shape=(8, 32, 32),
        n_actions=45,
        gamma=0.99, gae_lambda=0.95,
        device=torch.device("cpu"),
        flat_dim=289,
    )
    ppo = PPO(model=model, buffer=buf, device=torch.device("cpu"), use_amp=False)

    save_path = str(tmp_path / "test_hybrid_ckpt.pt")
    ppo.save(save_path)

    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    assert "obs_size" in ckpt
    assert ckpt["obs_size"] == 289
    assert "n_channels" in ckpt
    assert ckpt["n_channels"] == 8
    assert "grid_size" in ckpt
    assert ckpt["grid_size"] == 32


def test_config_accepts_hybrid():
    from rl.config import Config
    cfg = Config(obs_mode="hybrid")
    assert cfg.obs_mode == "hybrid"


def test_config_rejects_invalid_obs_mode():
    from rl.config import Config
    with pytest.raises(ValueError):
        Config(obs_mode="bogus")


def test_hybrid_buffer_flat_storage():
    from rl.rollout_buffer import _TensorRolloutBuffer

    buf = _TensorRolloutBuffer(
        n_steps=4, n_envs=2,
        obs_shape=(8, 32, 32),
        n_actions=45,
        gamma=0.99, gae_lambda=0.95,
        device=torch.device("cpu"),
        flat_dim=289,
    )
    assert buf.flat_obs is not None
    assert buf.flat_obs.shape == (8, 289)

    flat = torch.randn(2, 289)
    mm = torch.randn(2, 8, 32, 32)
    buf.add(
        obs=mm,
        action=torch.zeros(2, dtype=torch.long),
        reward=torch.ones(2),
        log_prob=torch.zeros(2),
        value=torch.zeros(2),
        done=torch.zeros(2, dtype=torch.bool),
        terminated=torch.zeros(2, dtype=torch.bool),
        flat=flat,
    )
    assert buf.flat_obs[0:2].shape == (2, 289)
    assert torch.allclose(buf.flat_obs[0:2], flat)


def test_hybrid_buffer_no_flat():
    from rl.rollout_buffer import _TensorRolloutBuffer

    buf = _TensorRolloutBuffer(
        n_steps=4, n_envs=2,
        obs_shape=(8, 32, 32),
        n_actions=45,
        gamma=0.99, gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    assert buf.flat_obs is None
    batches = list(buf.get_batches(4))
    for batch in batches:
        assert "flat" not in batch


def test_hybrid_balanced_branches_contract():
    """Ребаланс 2026-09-22: ветки уравновешены, flatten CNN не топит flat.

    Контракт: CNN после адаптивного пула 2×2 (64*2*2=256) проецируется в
    ширину flat-ствола; joint принимает 512 (=256+256), а не 65792; после
    joint-линейного слоя стоит LayerNorm (выравнивает масштабы ветвей).
    """
    import torch.nn as nn

    from rl.actor_critic_hybrid import ActorCriticHybrid

    m = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[256, 256], device="cpu",
    )
    assert m.cnn_head[0].out_features == 256
    assert m.flat_trunk[0].out_features == 256
    assert m.joint[0].in_features == 512
    assert m.joint[0].out_features == 256
    assert any(isinstance(mod, nn.LayerNorm) for mod in m.joint)


def test_hybrid_joint_weight_energy_parity():
    """Анти-разбавление: энергия flat-колонок joint-веса сопоставима с CNN.

    На старой архитектуре плоские колонки (256 из 65792) несли ~0.3%
    энергии — «плоские подсказки» тонули. Равные ширины ветвей + общий
    orthogonal init дают паритет порядка единицы; вываливание за [0.5, 2.0]
    означает возврат к перекосу.
    """
    import torch

    torch.manual_seed(0)
    from rl.actor_critic_hybrid import ActorCriticHybrid

    m = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[256, 256], device="cpu",
    )
    w = m.joint[0].weight            # [256, 512]
    flat_part = w[:, :256].norm().item()
    cnn_part = w[:, 256:].norm().item()
    ratio = flat_part / max(cnn_part, 1e-12)
    assert 0.5 <= ratio <= 2.0, f"branch energy ratio {ratio:.3f} out of parity"


def test_hybrid_cnn_head_grid_independent():
    """Ширина ветвей не зависит от размера миникарты (пул адаптивный)."""
    from rl.actor_critic_hybrid import ActorCriticHybrid

    for grid in (32, 33, 57):
        m = ActorCriticHybrid(
            obs_size=289, n_channels=8, grid_size=grid,
            n_actions=45, hidden_sizes=[256, 256], device="cpu",
        )
        assert m.cnn_head[0].in_features == 256
        assert m.joint[0].in_features == 512
        flat = torch.randn(2, 289)
        mm = torch.randn(2, 8, grid, grid)
        logits, values = m(flat, mm)
        assert logits.shape == (2, 45)
        assert values.shape == (2, 1)


def test_hybrid_actor_mask_proj_zero_init_is_noop(hybrid_model):
    """Variant B: W=0 ⇒ logits identical with/without masks at init."""
    flat = torch.randn(3, 289)
    mm = torch.randn(3, 8, 32, 32)
    masks = torch.ones(3, 45)
    masks[:, 4] = 0.0
    with torch.no_grad():
        logits_a, _ = hybrid_model(flat, mm)
        logits_b, _ = hybrid_model(flat, mm, action_masks=masks)
    assert torch.allclose(logits_a, logits_b)
    assert hybrid_model.actor_mask_proj.weight.abs().sum().item() == 0.0
