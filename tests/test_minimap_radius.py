import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from cpp_env import CppColonyEnv


def test_set_minimap_radius_single():
    env = CppColonyEnv(map_size=100)
    try:
        int(env.cpp_env.minimap_radius())  # API геттера на месте
        mm0 = env.cpp_env.minimap()
        assert mm0.shape == (8, 32, 32)

        env.cpp_env.set_minimap_radius(5)
        mm5 = env.cpp_env.minimap()
        assert mm5.shape == (8, 32, 32)
    finally:
        env.close()


def test_set_minimap_radius_vec():
    from cpp_vecenv import CppVecEnv
    venv = CppVecEnv(n_envs=2, map_size=100)
    try:
        venv.venv.set_minimap_radius(3)
        mm = venv.venv.minimap_batch()
        assert mm.shape == (2, 8, 32, 32)
    finally:
        venv.close()
