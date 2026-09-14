import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_normalize():
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=5)
    # Set known stats
    norm._rms.set_mean([0.0, 0.0, 0.0, 0.0, 0.0])
    norm._rms.set_var([1.0, 1.0, 1.0, 1.0, 1.0])

    obs = np.array([1.0, -1.0, 2.0, 0.0, 0.5], dtype=np.float32)
    result = norm.normalize(obs)

    # With mean=0, var=1: normalize = (v - 0) / sqrt(1 + eps) ≈ v
    np.testing.assert_allclose(result, [1.0, -1.0, 2.0, 0.0, 0.5], atol=0.01)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_save_load(tmp_path):
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=3)
    norm._rms.set_mean([1.0, 2.0, 3.0])
    norm._rms.set_var([0.5, 0.5, 0.5])
    norm._rms.set_count(100)

    path = tmp_path / "norm.json"
    norm.save(str(path))

    norm2 = Normalizer(obs_size=3)
    norm2.load(str(path))

    np.testing.assert_allclose(norm2._rms.mean(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(norm2._rms.var(), [0.5, 0.5, 0.5])
    assert norm2._rms.count() == 100


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_to_from_dict():
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=4)
    norm._rms.set_mean([1.0, 2.0, 3.0, 4.0])
    norm._rms.set_var([1.0, 1.0, 1.0, 1.0])
    norm._rms.set_count(50)

    d = norm.to_dict()
    assert d["mean"] == [1.0, 2.0, 3.0, 4.0]
    assert d["var"] == [1.0, 1.0, 1.0, 1.0]
    assert d["count"] == 50

    norm2 = Normalizer.from_dict(d)
    np.testing.assert_allclose(norm2._rms.mean(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(norm2._rms.var(), [1.0, 1.0, 1.0, 1.0])
    assert norm2._rms.count() == 50


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_set_update():
    """Verify set_update(False) prevents statistics updates."""
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=3)
    norm._rms.set_mean([0.0, 0.0, 0.0])
    norm._rms.set_var([1.0, 1.0, 1.0])

    # Enable updates, update stats
    norm.set_update(True)
    norm.update(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    mean_after_update = list(norm._rms.mean())

    # Disable updates, try to update again
    norm.set_update(False)
    norm.update(np.array([100.0, 200.0, 300.0], dtype=np.float32))
    mean_after_disabled = list(norm._rms.mean())

    # Mean should not have changed after disabling updates
    np.testing.assert_allclose(mean_after_update, mean_after_disabled)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_load_cpp_nested_shape(tmp_path):
    """Verify Normalizer.load reads the C++ save_normalization nested shape.

    C++ ColonyVecEnvCpp::save_normalization writes:
        {"obs_rms": {"mean": [...], "var": [...], "count": N},
         "norm_obs": true, "norm_reward": true,
         "clip_obs": 10.0, "clip_reward": 10.0}
    Python Normalizer.load must accept this shape (stats under "obs_rms").
    """
    import json
    from cpp_env import Normalizer

    mean = [0.1, 0.2, 0.3]
    var = [1.1, 2.2, 3.3]
    count = 42

    cpp_shape = {
        "obs_rms": {"mean": mean, "var": var, "count": count},
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
        "clip_reward": 10.0,
    }
    path = tmp_path / "cpp_norm.json"
    with open(path, "w") as f:
        json.dump(cpp_shape, f)

    norm = Normalizer(obs_size=3)
    norm.load(str(path))

    np.testing.assert_allclose(norm._rms.mean(), mean, rtol=1e-5)
    np.testing.assert_allclose(norm._rms.var(), var, rtol=1e-5)
    assert norm._rms.count() == count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_load_rejects_obs_size_mismatch(tmp_path):
    """A v1 (287) file on a v0 (246) normalizer must fail, not misalign."""
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=287)
    norm._rms.set_mean([0.0] * 287)
    norm._rms.set_var([1.0] * 287)
    path = tmp_path / "norm.json"
    norm.save(str(path))

    with pytest.raises(ValueError) as excinfo:
        Normalizer(obs_size=246).load(str(path))
    assert "normalization mismatch" in str(excinfo.value)
    assert "287" in str(excinfo.value)
    # matching size still loads
    Normalizer(obs_size=287).load(str(path))
