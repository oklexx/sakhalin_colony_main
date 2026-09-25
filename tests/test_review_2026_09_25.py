"""Регрессы код-ревью 2026-09-25 (P0/P1).

Каждый тест пинит конкретный багфикс; детали — в комментарии над тестом.
Тесты, которым нужен только torch/numpy, работают везде; GUI-тест скипается
без PyQt6, но выполняется в CI.
"""
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from rl.actor_critic import ActorCritic  # noqa: E402
from rl.actor_critic_cnn import ActorCriticCNN  # noqa: E402
from rl.actor_critic_hybrid import ActorCriticHybrid  # noqa: E402
from rl.config import Config  # noqa: E402
from rl.curriculum import CurriculumState, ckpt_flat_width  # noqa: E402
from rl.env_manager import _make_ppo  # noqa: E402
from rl.rollout_buffer import RolloutBuffer  # noqa: E402
from train_ui2 import protocol as P  # noqa: E402
from train_ui2.evaluator import (  # noqa: E402
    _load_policy,
    _resolve_difficulty,
    _resolve_tax_to_debt,
)

DEVICE = torch.device("cpu")


# ── P0: ckpt_flat_width врал для CNN ─────────────────────────────────────────
# `trunk.0.weight` есть и у ActorCriticCNN (вход после свёртки, 64·G·G), и его
# ширина 65536 сравнивалась с obs среды → resume minimap-моделей падал всегда.

def test_ckpt_width_mlp_reports_flat_input():
    model = ActorCritic(obs_size=18, n_actions=5, hidden_sizes=[8], device=DEVICE)
    assert ckpt_flat_width(model.state_dict()) == 18


def test_ckpt_width_cnn_only_is_none():
    model = ActorCriticCNN(n_channels=8, grid_size=32, n_actions=5,
                           hidden_sizes=[8], device=DEVICE)
    assert "trunk.0.weight" in model.state_dict()  # ловушка старого кода
    assert ckpt_flat_width(model.state_dict()) is None


def test_ckpt_width_hybrid_reports_flat_input():
    model = ActorCriticHybrid(obs_size=18, n_channels=8, grid_size=32,
                              n_actions=5, hidden_sizes=[8], device=DEVICE)
    assert ckpt_flat_width(model.state_dict()) == 18


def test_ckpt_width_non_mapping_is_none():
    assert ckpt_flat_width(object()) is None
    assert ckpt_flat_width({}) is None


# ── P0: _make_ppo недооценивал число шагов оптимизатора ──────────────────────
# get_batches() отдаёт и неполный последний батч, а floor давал 3 вместо 4 —
# косинус LR достигал пола раньше конца обучения.

def _ppo_for_steps(n_steps=8, n_envs=2, batch_size=10, n_epochs=2,
                   total_timesteps=16):
    cfg = Config(n_steps=n_steps, n_envs=n_envs, batch_size=batch_size,
                 n_epochs=n_epochs, total_timesteps=total_timesteps,
                 use_amp=False, torch_compile=False)
    model = ActorCritic(4, 3, [8], DEVICE)
    buf = RolloutBuffer(n_steps, n_envs, 4, 3, 0.99, 0.95, DEVICE)
    return _make_ppo(cfg, model, buf)


def test_make_ppo_counts_partial_last_batch():
    ppo = _ppo_for_steps()
    # 16 сэмплов / батч 10 → 2 батча на эпоху, 1 роллаут, 2 эпохи.
    assert ppo._total_training_steps == 1 * 2 * 2


def test_make_ppo_exact_division_unchanged():
    ppo = _ppo_for_steps(batch_size=8)
    assert ppo._total_training_steps == 1 * 2 * 2
    ppo = _ppo_for_steps(batch_size=16)
    assert ppo._total_training_steps == 1 * 2 * 1


# ── P0: parity через assert умирал под `python -O` ────────────────────────────

def _stub_env_manager(curriculum: dict):
    from rl.env_manager import EnvManager
    em = EnvManager.__new__(EnvManager)
    em.vec_env = SimpleNamespace(
        venv=SimpleNamespace(curriculum=lambda: dict(curriculum)))
    return em


def test_parity_mismatch_raises_runtime_error_not_assert():
    st = CurriculumState(False, ("Goldmine",), True, (1.0,) * 9, 0)
    env_report = st.to_dict()
    env_report["allowed_builds"] = ["City"]  # расхождение со средой
    em = _stub_env_manager(env_report)
    with pytest.raises(RuntimeError, match="курикулум не применён"):
        em._assert_curriculum_parity(st)


def test_parity_match_passes():
    st = CurriculumState.all()
    em = _stub_env_manager(st.to_dict())
    em._assert_curriculum_parity(st)  # не должно падать


def test_parity_contains_no_assert_statements():
    """Статическая гарантия: `python -O` не может выключить проверку."""
    src = (REPO / "rl" / "env_manager.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_curriculum_parity":
            found = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
            assert not found, "parity must not use assert (dies under -O)"
            return
    raise AssertionError("_assert_curriculum_parity not found")


# ── P0: watch_champion определял CNN через hasattr "cnn" ─────────────────────
# У ActorCriticCNN свёртка называется `conv` → чистая minimap-модель шла по
# flat-ветке и падала в Conv2d с 2-D входом.

@pytest.fixture(scope="module")
def wc():
    """Модуль watch_champion: с настоящим colony_cpp, а без него — со стабом.

    Стаб живёт только на время этого модуля и убирается за собой: иначе он
    протекает в sys.modules и ломает тесты, которые честно скипаются без
    расширения (паттерн — из tests/test_watch_visual.py).
    """
    saved = {n: sys.modules.get(n)
             for n in ("colony_cpp", "cpp_env", "watch_champion")}
    stubbed = False
    try:
        import watch_champion as mod
    except ImportError:
        sys.modules["colony_cpp"] = SimpleNamespace()
        for name in ("cpp_env", "watch_champion"):
            sys.modules.pop(name, None)
        try:
            import watch_champion as mod
        except ImportError as e:
            pytest.skip(f"watch_champion не импортируется даже со стабом: {e}")
        stubbed = True
    yield mod
    if stubbed:
        for name in ("colony_cpp", "cpp_env", "watch_champion"):
            sys.modules.pop(name, None)
        for name, old_mod in saved.items():
            if old_mod is not None:
                sys.modules[name] = old_mod


def test_policy_kinds_cnn(wc):
    cnn = ActorCriticCNN(n_channels=8, grid_size=32, n_actions=5,
                         hidden_sizes=[8], device=DEVICE)
    assert wc._policy_kinds(cnn) == (False, True)


def test_policy_kinds_hybrid(wc):
    hybrid = ActorCriticHybrid(obs_size=8, n_channels=8, grid_size=32,
                               n_actions=5, hidden_sizes=[8], device=DEVICE)
    assert wc._policy_kinds(hybrid) == (True, False)


def test_policy_kinds_mlp(wc):
    mlp = ActorCritic(obs_size=8, n_actions=5, hidden_sizes=[8], device=DEVICE)
    assert wc._policy_kinds(mlp) == (False, False)


# ── P1: evaluator восстанавливает difficulty/tax из меты ─────────────────────
# Старая проверка difficulty срабатывала после слияния меты и была всегда
# ложна → eval без явного difficulty молча играл на "normal". А
# meta.get("config", {}) падало, если "config" в мете был null.

def test_resolve_difficulty_explicit_wins():
    assert _resolve_difficulty({"difficulty": "light"}, "hard") == "hard"


def test_resolve_difficulty_from_meta():
    assert _resolve_difficulty({"difficulty": "light"}, None) == "light"


def test_resolve_difficulty_default_normal():
    assert _resolve_difficulty({}, None) == "normal"
    assert _resolve_difficulty({"difficulty": None}, None) == "normal"


def test_resolve_tax_explicit_wins():
    assert _resolve_tax_to_debt({"tax_to_debt": True}, False) is False


def test_resolve_tax_from_meta_and_config():
    assert _resolve_tax_to_debt({"tax_to_debt": False}, None) is False
    assert _resolve_tax_to_debt({"config": {"tax_to_debt": False}}, None) is False
    assert _resolve_tax_to_debt({}, None) is True


def test_resolve_tax_null_config_no_crash():
    assert _resolve_tax_to_debt({"config": None}, None) is True
    assert _resolve_tax_to_debt({"config": "garbage"}, None) is True


# ── P1: infer hidden_sizes цеплял веса LayerNorm ─────────────────────────────
# У гибрида joint.* содержит 1-D веса LayerNorm с тем же суффиксом .weight —
# без фильтра слои дублировались ([16, 8, 8] вместо [16, 8]) и _load_policy
# падал на несовпадении архитектуры.

def test_load_policy_hybrid_without_hidden_sizes(tmp_path):
    model = ActorCriticHybrid(obs_size=18, n_channels=8, grid_size=32,
                              n_actions=5, hidden_sizes=[16, 8], device=DEVICE)
    ckpt_path = tmp_path / "hybrid.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    loaded = _load_policy(ckpt_path, DEVICE)
    assert isinstance(loaded, ActorCriticHybrid)
    # Веса реально встали, а не остались случайными.
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_load_policy_cnn_without_hidden_sizes(tmp_path):
    model = ActorCriticCNN(n_channels=8, grid_size=32, n_actions=5,
                           hidden_sizes=[16, 8], device=DEVICE)
    ckpt_path = tmp_path / "cnn.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    loaded = _load_policy(ckpt_path, DEVICE)
    assert isinstance(loaded, ActorCriticCNN)


# ── P1: protocol — контракт ValueError и JSON-безопасность ───────────────────

def test_decode_command_missing_cmd_is_value_error():
    with pytest.raises(ValueError, match="missing fields"):
        P.decode('{"type": "command", "payload": {}}')


def test_decode_command_non_dict_payload_is_value_error():
    with pytest.raises(ValueError, match="payload"):
        P.decode('{"type": "command", "cmd": "stop", "payload": [1, 2]}')


def test_decode_command_ok():
    msg = P.decode('{"type": "command", "cmd": "stop", "payload": {"x": 1}}')
    assert isinstance(msg, P.CommandMsg) and msg.cmd == "stop"


def test_encode_progress_top_actions_json_safe():
    # encode идёт с allow_nan=False — без sanitize в to_dict inf/nan роняли
    # сериализацию прогресса целиком.
    msg = P.ProgressMsg(done=1, total=10,
                        top_actions={"a": math.inf, "b": math.nan})
    payload = json.loads(P.encode(msg))
    assert payload["top_actions"] == {"a": 0.0, "b": 0.0}


def test_decode_progress_non_dict_maps_no_crash():
    msg = P.decode('{"type": "progress", "done": 1, "total": 10,'
                   ' "top_actions": [1, 2], "action_counts": "x"}')
    assert msg.top_actions == {}
    assert msg.action_counts == {}


# ── P1: worker.py — shebang/docstring до импортов ─────────────────────────────

def test_worker_module_docstring_first():
    """shebang + docstring обязаны идти до кода (иначе SyntaxWarning/eager help)."""
    src = (REPO / "train_ui2" / "worker.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    assert lines[0].startswith("#!"), "shebang must be the first line"
    assert ast.get_docstring(ast.parse(src), clean=False) is not None
    assert ast.get_docstring(ast.parse(src)).startswith("Worker process")


# ── P0: UI больше не форсирует gamma/ent_coef при загрузке ───────────────────
# Удалены безусловные «миграции» gamma→0.99999 и ent_coef→0.01: дефолт gamma
# давно 0.999, а миграция переписывала его при КАЖДОЙ загрузке.

def test_load_state_preserves_gamma_and_ent_coef(tmp_path, monkeypatch):
    pytest.importorskip("PyQt6.QtWidgets")
    import train_ui2.main_window as mw

    state = {"gamma": 0.999, "ent_coef": 0.05, "config_version": 99}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(mw, "CONFIG_PATH", cfg_file)
    loaded = mw.MainWindow._load_state(object())
    assert loaded["gamma"] == 0.999
    assert loaded["ent_coef"] == 0.05
