import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_ui2.parameter_widget import FLOAT_MIN, INT_MIN, PARAM_SPECS, ParamSpec, scale_value, spec_for


def test_scale_int_double():
    assert scale_value(10, 2, is_int=True) == 20


def test_scale_int_half():
    assert scale_value(10, 0.5, is_int=True) == 5


def test_scale_int_rounding():
    assert scale_value(5, 0.5, is_int=True) == 2  # 2.5 rounds to 2 (banker's) or 3; accept both
    assert scale_value(3, 0.5, is_int=True) in (1, 2)


def test_scale_int_min_clamp():
    assert scale_value(1, 0.5, is_int=True) == INT_MIN
    assert scale_value(1, 0.1, is_int=True) == INT_MIN


def test_scale_float_double():
    assert scale_value(0.001, 2, is_int=False) == 0.002


def test_scale_float_half():
    assert scale_value(0.001, 0.5, is_int=False) == 0.0005


def test_scale_float_min_clamp():
    assert scale_value(1e-10, 0.5, is_int=False) == FLOAT_MIN


def test_scale_float_rounding():
    assert scale_value(0.123456789, 2, is_int=False) == 0.24691358


def test_scale_zero_factor():
    with pytest.raises(ValueError):
        scale_value(10, 0, is_int=True)
    with pytest.raises(ValueError):
        scale_value(0.1, 0, is_int=False)


def test_scale_negative_factor():
    assert scale_value(10, -2, is_int=True) == INT_MIN
    assert scale_value(0.1, -0.5, is_int=False) == FLOAT_MIN


def test_specs_keys():
    keys = [s.key for s in PARAM_SPECS]
    expected = ["total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs",
                "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
                "vf_coef", "max_grad_norm", "n_layers", "seed", "map_size",
                "target_kl", "save_freq", "eval_freq", "eval_episodes",
                "eval_min_days", "eval_min_bases", "early_stopping_patience"]
    assert keys == expected


def test_specs_fields():
    for s in PARAM_SPECS:
        assert s.min >= 0
        assert s.max >= s.min
        assert s.default >= s.min
        assert s.default <= s.max
        assert len(s.tooltip) > 10


def test_spec_for_lookup():
    s = spec_for("learning_rate")
    assert s.key == "learning_rate"
    assert s.is_int is False


def test_spec_for_unknown():
    with pytest.raises(KeyError):
        spec_for("nope")


def test_param_spec_dataclass():
    s = ParamSpec("k", "Л", True, 1, 10, 5, "tip", step=2.0, decimals=2)
    assert s.step == 2.0
    assert s.decimals == 2
