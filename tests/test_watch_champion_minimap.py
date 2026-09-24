import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def test_hybrid_model_act_with_minimap():
    """Verify hybrid model can act with both flat obs and minimap."""
    from rl.actor_critic_hybrid import ActorCriticHybrid
    model = ActorCriticHybrid(
        obs_size=289, n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[128, 128], device="cpu",
    )
    model.eval()
    flat = torch.randn(289)
    mm = torch.randn(8, 32, 32)
    action, log_prob, value = model.act(
        flat.unsqueeze(0), mm.unsqueeze(0), deterministic=True
    )
    assert 0 <= int(action.item()) < 45


def test_cnn_model_act_with_minimap():
    """Verify CNN model can act with minimap."""
    from rl.actor_critic_cnn import ActorCriticCNN
    model = ActorCriticCNN(
        n_channels=8, grid_size=32,
        n_actions=45, hidden_sizes=[128, 128], device="cpu",
    )
    model.eval()
    mm = torch.randn(8, 32, 32)
    logits, _ = model(mm.unsqueeze(0))
    action = int(logits.argmax(dim=-1).item())
    assert 0 <= action < 45
