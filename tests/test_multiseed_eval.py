import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config


def test_eval_seeds_default():
    cfg = Config(n_envs=2, n_steps=5, map_size=100)
    assert cfg.eval_seeds == [42]
    assert cfg.eval_use_median is True


def test_eval_seeds_custom():
    cfg = Config(n_envs=2, n_steps=5, map_size=100)
    cfg.eval_seeds = [1, 2, 3]
    cfg.eval_use_median = False
    assert cfg.eval_seeds == [1, 2, 3]
    assert cfg.eval_use_median is False


def test_eval_seeds_parsing():
    """Verify argparse-style parsing works."""
    import shlex
    args = shlex.split("--eval-seeds 42 43 44")
    seeds = []
    i = 1
    while i < len(args):
        seeds.append(int(args[i]))
        i += 1
    assert seeds == [42, 43, 44]
