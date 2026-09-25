import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config


def test_config_accepts_minimap_radius():
    cfg = Config(n_envs=2, n_steps=5, map_size=100)
    cfg.minimap_radius = 28
    assert cfg.minimap_radius == 28


def test_config_accepts_hybrid_obs_mode():
    cfg = Config(n_envs=2, n_steps=5, map_size=100)
    cfg.obs_mode = "hybrid"
    assert cfg.obs_mode == "hybrid"
