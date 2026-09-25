"""Regression: the critic bootstrap must respect obs_mode (баг «tuple в Linear»).

Симптом (2026-09-20, старт обучения в hybrid-режиме)::

    File "rl/async_trainer.py", line 202, in _collect_rollout
        last_value = (self.em.ppo.model.get_value(obs, last_masks) ...
    File "rl/actor_critic_hybrid.py", line 108, in forward
        h_flat = self.flat_trunk(flat_in)
    TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple

Причина: режим наблюдения читался как `getattr(self.em, "obs_mode", "flat")`,
но `EnvManager` хранил его только в `cfg` → гибрид всегда попадал в flat-ветку,
где пара `(flat, minimap)` уходила в политику одним «тензором», а маска действий
— вторым позиционным аргументом (у гибрида это `minimap`).

Тесты гоняют настоящий `AsyncTrainer._collect_rollout` на подставном
EnvManager'е (без colony_cpp), поэтому падение воспроизводится дословно.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.actor_critic import ActorCritic
from rl.actor_critic_cnn import ActorCriticCNN
from rl.actor_critic_hybrid import ActorCriticHybrid
from rl.async_trainer import AsyncTrainer
from rl.config import Config
from rl.ppo import PPO
from rl.rollout_buffer import RolloutBuffer, _TensorRolloutBuffer

N_ENVS = 2
N_STEPS = 3
FLAT_DIM = 12
GRID = 4            # маленькая миникарта: тесты не должны считать 32×32
CH = 8
N_ACTIONS = 5
DEVICE = torch.device("cpu")


class _FakeVecEnv:
    """Минимум поверхности CppVecEnv, который трогает _collect_rollout."""

    def __init__(self, n_envs=N_ENVS, n_actions=N_ACTIONS):
        self.num_envs = n_envs
        self.action_masks = np.ones((n_envs, n_actions), dtype=np.float32)

    def close(self):
        pass


class _FakeEnvManager:
    """EnvManager-двойник для obs_mode из конструктора."""

    def __init__(self, obs_mode: str = "flat", expose_obs_mode: bool = True):
        self.obs_mode_value = obs_mode
        self.n_envs = N_ENVS
        self.device = DEVICE
        self.vec_env = _FakeVecEnv()
        self.cfg = types.SimpleNamespace(obs_mode=obs_mode, map_size=280)
        # Так делает настоящий EnvManager после фикса; флаг позволяет проверить,
        # что тренер больше не зависит от наличия атрибута (читает cfg).
        if expose_obs_mode:
            self.obs_mode = obs_mode

        torch.manual_seed(0)
        if obs_mode == "hybrid":
            self.model = ActorCriticHybrid(
                obs_size=FLAT_DIM, n_channels=CH, grid_size=GRID,
                n_actions=N_ACTIONS, hidden_sizes=[8, 8], device=DEVICE,
            )
            self.buffer = _TensorRolloutBuffer(
                n_steps=N_STEPS, n_envs=N_ENVS, obs_shape=(CH, GRID, GRID),
                n_actions=N_ACTIONS, gamma=0.99, gae_lambda=0.95,
                device=DEVICE, flat_dim=FLAT_DIM,
            )
        elif obs_mode == "minimap":
            self.model = ActorCriticCNN(
                n_channels=CH, grid_size=GRID, n_actions=N_ACTIONS,
                hidden_sizes=[8], device=DEVICE,
            )
            self.buffer = _TensorRolloutBuffer(
                n_steps=N_STEPS, n_envs=N_ENVS, obs_shape=(CH, GRID, GRID),
                n_actions=N_ACTIONS, gamma=0.99, gae_lambda=0.95, device=DEVICE,
            )
        else:
            self.model = ActorCritic(FLAT_DIM, N_ACTIONS, [8], DEVICE)
            self.buffer = RolloutBuffer(
                n_steps=N_STEPS, n_envs=N_ENVS, obs_size=FLAT_DIM,
                n_actions=N_ACTIONS, gamma=0.99, gae_lambda=0.95, device=DEVICE,
            )
        self.ppo = PPO(model=self.model, buffer=self.buffer, device=DEVICE,
                       use_amp=False, n_epochs=1, batch_size=N_ENVS)

    @property
    def action_names(self):
        return [f"A{i}" for i in range(N_ACTIONS)]

    def _new_obs(self):
        if self.obs_mode_value == "hybrid":
            return (torch.randn(N_ENVS, FLAT_DIM),
                    torch.randn(N_ENVS, CH, GRID, GRID))
        if self.obs_mode_value == "minimap":
            return torch.randn(N_ENVS, CH, GRID, GRID)
        return torch.randn(N_ENVS, FLAT_DIM)

    def collect_step(self, obs):
        masks = torch.as_tensor(self.vec_env.action_masks, dtype=torch.float32)
        if self.obs_mode_value == "hybrid":
            flat, minimap = obs
            sampled = self.ppo.collect_step(flat, minimap=minimap, action_masks=masks)
            prev = minimap
        else:
            sampled = self.ppo.collect_step(obs, action_masks=masks)
            prev = obs
        new_obs = self._new_obs()
        rewards = torch.randn(N_ENVS)
        dones = torch.zeros(N_ENVS, dtype=torch.bool)
        kwargs = {}
        if self.obs_mode_value == "hybrid":
            kwargs["flat"] = obs[0]
        self.buffer.add(prev, sampled["action"], rewards, sampled["log_prob"],
                        sampled["value"], dones, terminated=dones,
                        action_masks=masks, **kwargs)
        return new_obs, [{} for _ in range(N_ENVS)]

    def close(self):
        pass


def _trainer(obs_mode: str, expose_obs_mode: bool = True) -> AsyncTrainer:
    cfg = Config(n_envs=N_ENVS, n_steps=N_STEPS, obs_mode=obs_mode,
                 use_amp=False, save_freq=0, eval_freq=0)
    return AsyncTrainer(cfg=cfg, env_manager=_FakeEnvManager(obs_mode, expose_obs_mode))


# ── сам регресс ──────────────────────────────────────────────────────────────

def test_hybrid_collect_rollout_bootstraps_value():
    """Первый же rollout гибрида обязан считать V(s_last), а не падать в Linear.

    `expose_obs_mode=False` — ровно как в настоящем EnvManager до фикса: атрибута
    нет, режим живёт только в `cfg`.
    """
    trainer = _trainer("hybrid", expose_obs_mode=False)
    obs = trainer.em._new_obs()

    rollout = trainer._collect_rollout(obs)

    assert rollout["last_value"].shape == (N_ENVS,)
    assert np.isfinite(rollout["last_value"]).all()
    assert rollout["last_done"].shape == (N_ENVS,)
    # финальное наблюдение возвращается тем же форматом, что дал env
    assert isinstance(rollout["final_obs"], tuple) and len(rollout["final_obs"]) == 2


def test_hybrid_bootstrap_uses_action_masks():
    """Маска доступности уходит в critic_mask_proj, а не в слот миникарты."""
    trainer = _trainer("hybrid", expose_obs_mode=False)
    flat = torch.randn(N_ENVS, FLAT_DIM)
    mm = torch.randn(N_ENVS, CH, GRID, GRID)

    masks = torch.ones(N_ENVS, N_ACTIONS)
    with torch.no_grad():
        v_masked = trainer._bootstrap_value((flat, mm), masks)
        v_plain = trainer._bootstrap_value((flat, mm), None)
    assert v_masked.shape == (N_ENVS,)
    # critic_mask_proj инициализирован нулями → значения совпадают, но вызов
    # с маской обязан проходить (раньше он и был источником TypeError).
    assert torch.allclose(v_masked, v_plain)

    # ненулевая маска меняет значение → маска действительно дошла до критика
    torch.nn.init.constant_(trainer.em.model.critic_mask_proj.weight, 1.0)
    with torch.no_grad():
        v_on = trainer._bootstrap_value((flat, mm), masks)
        half = masks.clone()
        half[:, 0] = 0.0
        v_half = trainer._bootstrap_value((flat, mm), half)
    assert not torch.allclose(v_on, v_half)


def test_obs_mode_is_read_from_cfg_when_em_has_no_attribute():
    """EnvManager без `obs_mode` больше не означает «flat»."""
    trainer = _trainer("hybrid", expose_obs_mode=False)
    assert not hasattr(trainer.em, "obs_mode")
    assert trainer._obs_mode() == "hybrid"


def test_bootstrap_accepts_dict_obs():
    trainer = _trainer("hybrid", expose_obs_mode=False)
    obs = {"flat": torch.randn(N_ENVS, FLAT_DIM),
           "minimap": torch.randn(N_ENVS, CH, GRID, GRID)}
    value = trainer._bootstrap_value(obs, torch.ones(N_ENVS, N_ACTIONS))
    assert value.shape == (N_ENVS,)


def test_container_obs_on_flat_mode_fails_loudly():
    """Пара (flat, minimap) при obs_mode='flat' — ошибка с внятным текстом."""
    trainer = _trainer("flat", expose_obs_mode=False)
    trainer.cfg.obs_mode = "flat"
    obs = (torch.randn(N_ENVS, FLAT_DIM), torch.randn(N_ENVS, CH, GRID, GRID))
    with pytest.raises(TypeError, match="obs_mode='flat'"):
        trainer._bootstrap_value(obs, None)


# ── остальные режимы не сломаны ──────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["flat", "minimap"])
def test_flat_and_minimap_rollouts_still_work(mode):
    trainer = _trainer(mode, expose_obs_mode=False)
    obs = trainer.em._new_obs()
    rollout = trainer._collect_rollout(obs)
    assert rollout["last_value"].shape == (N_ENVS,)
    assert np.isfinite(rollout["last_value"]).all()


# ── контракт EnvManager ──────────────────────────────────────────────────────

def test_env_manager_exposes_obs_mode(monkeypatch):
    """Настоящий EnvManager публикует obs_mode (его и читал тренер)."""
    import rl.env_manager as em_mod

    monkeypatch.setitem(sys.modules, "colony_cpp_api",
                        types.SimpleNamespace(require_colony=lambda **kw: {"version": 99}))

    class _Vec:
        num_envs = N_ENVS
        obs_size = FLAT_DIM
        action_space = types.SimpleNamespace(n=N_ACTIONS)
        action_masks = np.ones((N_ENVS, N_ACTIONS), dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr(em_mod, "_make_vec_env", lambda cfg: _Vec())

    cfg = Config(obs_mode="hybrid", n_envs=N_ENVS)
    em = em_mod.EnvManager(cfg, DEVICE)
    try:
        assert em.obs_mode == "hybrid"
        assert em.cfg.obs_mode == em.obs_mode
    finally:
        em.close()


@pytest.mark.parametrize("rel_path", ["rl/async_trainer.py", "watch_champion.py"])
def test_module_global_names_resolve(rel_path: str) -> None:
    """Каждое глобальное имя в модуле должно существовать (импорт/присваивание).

    Историческая ловушка №1 (`rl/async_trainer.py`): `json.dump(...)` при записи
    мета-чекпойнта стоял внутри `try: ... except Exception: pass`, а `json` был
    импортирован локально в ДРУГИХ методах. NameError проглатывался → чекпойнт
    оставался без `.meta.json`, и watch/турнир не могли определить стадию
    курикулума, под которой он учился. В турнире то же самое делал
    несуществующий `norm_str`.

    Ловушка №2 (`watch_champion.py`): `except (json.JSONDecodeError, OSError)`
    в main() при локальном `import json as _json` — битый `best_model.meta.json`
    давал NameError ВМЕСТО аккуратного пропуска, то есть наблюдение падало ровно
    в той ситуации, где должно было продолжать работу.

    Тихие падения не видны ни в логе, ни в тестах, поэтому проверяем связность
    имён статически (symtable, без импорта модуля — torch/colony_cpp не нужны).
    """
    import ast
    import builtins
    import symtable

    src_path = Path(__file__).resolve().parents[1] / rel_path
    src = src_path.read_text(encoding="utf-8")
    top = symtable.symtable(src, str(src_path), "exec")

    module_names = {sym.get_name() for sym in top.get_symbols()
                    if sym.is_assigned() or sym.is_imported() or sym.is_local()}
    # атрибуты модуля, которые подставляет импорт-машина, а не исходник
    module_names |= {"__file__", "__name__", "__doc__", "__package__",
                     "__loader__", "__spec__", "__builtins__", "__annotations__"}
    # имена, объявленные `global` внутри функций, считаются определёнными модулем
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Global):
            module_names.update(node.names)

    missing: list[str] = []

    def walk(table: "symtable.SymbolTable", where: str) -> None:
        for sym in table.get_symbols():
            name = sym.get_name()
            if sym.is_global() and name not in module_names and not hasattr(builtins, name):
                missing.append(f"{where}: {name}")
        for child in table.get_children():
            walk(child, f"{where}.{child.get_name()}")

    walk(top, src_path.name)
    assert not missing, f"неопределённые глобальные имена (NameError в рантайме): {missing}"


# ── статическая проверка «локальное имя читается до присваивания» ─────────────
#
# 2026-09-24 (`watch_champion.py`): в main() цикл headless-наблюдения читал
# `watch_temp` (1181), а присваивание стояло ниже (1231) — Python считает имя
# локальным на всю функцию, поэтому ЛЮБОЙ прогон без `--visual` падал с
# UnboundLocalError на первом шаге выбора действия. Тест выше (`globals`)
# такое не видит: имя-то локальное. Визуальные тесты тоже молчали — они всегда
# запускаются с `--visual`, где присваивание успевает выполниться.
#
# Проверка статическая (AST, без импорта модулей): для каждой функции собираем
# позиции чтений и записей имён в порядке исходника, игнорируя вложенные скоупы
# и объявления `global`/`nonlocal`. Если имя где-то присваивается и при этом
# читается РАНЬШЕ первого присваивания — это будущий UnboundLocalError.
# Проверка намеренно консервативная: присваивание в любой ветке считается
# присваиванием (ложных срабатываний нет).

_SCAN_DIRS = ("rl", "train_ui2", "python", "ui", "scripts")
_SCAN_ROOT_FILES = ("train.py", "training_lr.py", "watch_champion.py")

_NESTED_SCOPES = (
    "FunctionDef", "AsyncFunctionDef", "Lambda", "ClassDef",
    "ListComp", "SetComp", "DictComp", "GeneratorExp",
)


def _local_name_positions(fn) -> tuple[set, list, list]:
    """(параметры, чтения, записи) для одной функции, без вложенных скоупов."""
    import ast

    params = set()
    args = getattr(fn, "args", None)
    if args is not None:
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            params.add(a.arg)
        if args.vararg:
            params.add(args.vararg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)

    loads: list[tuple[tuple[int, int], str]] = []
    stores: list[tuple[tuple[int, int], str]] = []

    def pos(node) -> tuple[int, int]:
        return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))

    def walk(node) -> None:
        for child in ast.iter_child_nodes(node):
            if type(child).__name__ in _NESTED_SCOPES:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    stores.append((pos(child), child.name))
                continue  # тело — чужой скоуп, читает внешние имена законно
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Store):
                    stores.append((pos(child), child.id))
                elif isinstance(child.ctx, (ast.Load, ast.Del)):
                    loads.append((pos(child), child.id))
                continue
            if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                # `x += 1` — чтение и запись в одной точке
                loads.append((pos(child.target), child.target.id))
                stores.append((pos(child.target), child.target.id))
                walk(child.value)
                continue
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                continue
            walk(child)

    walk(fn)
    return params, loads, stores


def _read_before_assignment(src: str, path: str) -> list[str]:
    import ast

    tree = ast.parse(src)
    problems: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params, loads, stores = _local_name_positions(fn)
        declared_global = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                declared_global.update(node.names)
        first_store: dict[str, tuple[int, int]] = {}
        for p, name in stores:
            if name not in first_store or p < first_store[name]:
                first_store[name] = p
        for p, name in loads:
            if name in params or name in declared_global:
                continue
            store_pos = first_store.get(name)
            if store_pos is not None and store_pos > p:
                problems.append(
                    f"{path}:{p[0]}:{p[1]}: в {fn.name}() имя {name!r} читается до "
                    f"присваивания (единственное/первое присваивание — строка "
                    f"{store_pos[0]}) → UnboundLocalError"
                )
    return problems


def test_project_sources_have_no_local_read_before_assignment() -> None:
    """Ни одно локальное имя не читается раньше своего присваивания."""
    root = Path(__file__).resolve().parents[1]
    files: list[Path] = [root / name for name in _SCAN_ROOT_FILES]
    for d in _SCAN_DIRS:
        files.extend(sorted((root / d).rglob("*.py")))
    files = [f for f in files if f.is_file()]

    problems: list[str] = []
    for f in files:
        problems.extend(_read_before_assignment(f.read_text(encoding="utf-8"), str(f)))
    assert not problems, "чтение локального имени до присваивания:\n" + "\n".join(problems)
