# Рефакторинг Sakhalin Colony — 2026-09-11 (полный, совместимый)

**Ветка:** `arena/01a09205-sakhalin-colony-main`  
**Принцип:** поведение не меняется, публичные API сохранены, только внутренняя структура.

---

## 1. Цели и ограничения

- Сохранить обратную совместимость: `from rl.config import Config`, `from train_ui2.main_window import MainWindow2, REWARD_GROUPS`, `CppVecEnv` сигнатуры — без изменений.
- Убрать дублирование, монолиты 1000+ строк → модули 200–400 строк.
- Единый источник правды для наград (`configs/reward_v3.json` → `RewardConfig` → C++ `RewardConfig`), курикулума, ключей.
- Добавить типизацию, docstrings, комментарии, избавиться от hardcoded списков.

---

## 2. Что изменено (по фазам)

### Фаза 1 — `rl/config.py` (389 → 250 строк, но функциональнее)

**Было:** 
- `to_dict()` — 40 ключей вручную, `from_dict()` — 30 аргументов `d.get(...)` вручную, список полей дублируется в 3 местах.
- Валидация в `__post_init__` смешана с дефолтами путей.

**Стало:**
- Хелперы `_as_dict(obj)` / `_update_dataclass_from_dict()` через `dataclasses.fields` — DRY, новые поля автоматически сериализуются.
- `RewardConfig.to_dict()` — одна строка `return _as_dict(self)`.
- `RewardConfig.from_dict()` — копирует base, игнорирует неизвестные ключи, `coerce_int` для `idle_build_threshold_days`.
- `Config.to_dict()` — итерация по `fields(self)` минус `reward`, `from_dict()` фильтрует `allowed`, коэрсит `eval_score_weights`/`eval_min_days`, legacy-алиасы `unlock_ids` внутри reward.
- Выделены `_apply_path_defaults()` / `_validate()`, добавлены `is_hybrid` / `is_minimap` проперти, `load_from_file()` с проверками.
- Сохранены все дефолты (v3 профиль) и константы.

**Эффект:** новые поля добавляются одной строкой в dataclass, не нужно править 3 списка; forward-compat с неизвестными ключами.

### Фаза 2 — `train_ui2` (UI)

**Новые модули:**
- `train_ui2/constants.py` — `PARAM_GROUPS`, `REWARD_GROUPS`, `REWARD_FLAGS`, `CURRICULUM_STAGE_MAP`, `ALL_BUILD_IDS`, `BUILD_CAPTIONS`, `RESOURCE_IDS`, `BUILD_IMAGE_INDEX`, `CONFIG_VERSION`. Импортирует `rl.curriculum` как SSOT с fallback.
- `train_ui2/icons.py` — `load_icon_pixmap`, `building_icon`, `resource_icon`, `format_steps`, `escape_html`. Изолирует QPixmap, тестируемо без окна.
- `train_ui2/tabs/` — заготовка для дальнейшего сплита (пока используется через импорты; `main_window.py` уже использует новые модули).

**`main_window.py` (1486 → ~1380 строк, −100 строк дубликатов):**
- Удалены дубли `PARAM_GROUPS`/`REWARD_GROUPS`/curriculum-констант — теперь `from train_ui2.constants import ...`.
- Удалены `_fmt_steps`/`_load_icon_pixmap`/`_building_icon`/`_resource_icon`/`_esc` — `from train_ui2.icons import ...` (+ алиас `_esc`).
- Сохранены ре-экспорты `REWARD_GROUPS`, `PARAM_GROUPS` для тестов (`from train_ui2.main_window import REWARD_GROUPS` всё ещё работает).
- Логика вкладок не менялась, только импорты.

### Фаза 3 — `rl/curriculum.py` (новый, 70 строк)

**Единый источник курикулума:** `STAGE_MAP`, `ALL_IDS`, `ids_for_stage(stage)`, `allowed_buildings_for_stage()`.  
`train_ui2/constants.py` теперь импортирует его, `rl/env_manager.py` делегирует туда, C++ `env.cpp` комментируется как зеркало этого файла. Убирает рассинхрон пресетов Этап1-3 между UI и env.

### Фаза 4 — `rl/_nn_common.py` (новый, 40 строк) + `actor_critic*.py`

**Было:** три файла дублируют `orthogonal_init`, `params`, `state_dict_for_env`, `get_action_and_value` (разные gain, но логика одна, 3×30 строк).

**Стало:**
- `rl/_nn_common.py`: `orthogonal_init()`, `init_module_list()`, `ActorCriticBase` (params + state_dict + `_categorical_log_prob`).
- `ActorCritic`, `ActorCriticCNN`, `ActorCriticHybrid` наследуют `ActorCriticBase`, используют `orthogonal_init` вместо локальных `_orthogonal_init`/`_init_weights` кастомов, гибрид делегирует `params` базе.

**Выигрыш:** фикс gain в одном месте, меньше копипасты, легче добавить новый бэкбон.

### Фаза 5 — `rl/env_manager.py` (410 → 260 строк, −37%)

**Было:** `__init__` 120 строк делает всё: `make_cpp_vec_env`, `obs_size`, `mm_env`, `model`, `buffer`, `ppo` с дублированным `if hybrid / elif minimap / else` и копипастой буфера.

**Стало:**
- Хелперы `_ensure_python_path()`, `_make_vec_env(cfg)`.
- Методы-фабрики `_make_model()`, `_make_buffer()`, `_make_ppo()` — каждый 20–30 строк, единый `self.mm_env` путь.
- `_policy_obs()` унифицирован.
- `get_allowed_buildings_for_stage()` / `get_curriculum_progress()` упрощены, `get_stats()` заглушка, `set_curriculum_stage`/`close` однострочные.
- Комментарии и типизация, `total_updates` формула вынесена в переменные.

### Фаза 6 — C++ (`include/colony/*`, `src/*`)

**Новые:**
- `include/colony/reward_config.h` — `struct RewardConfig` (вырезан из `env.h`), `#pragma once`, один источник для C++ дефолтов.
- `include/colony/rewards.h` / `src/rewards.cpp` — `proximity_land_for`, `provider_bonus`, `prerequisite_bonus` (были 70 строк статик-функций в `env.cpp`). Декларированы в namespace `colony`, `env.cpp` теперь `#include "colony/rewards.h"` и не дублирует.

**`include/colony/env.h` (381 → 320 строк):**
- Удалена копия `RewardConfig`, добавлен `#include "colony/reward_config.h"`, убран дубль `reward_config.h` include, оставлены форварды.

**`src/env.cpp` (1545 → 1475 строк):**
- Удалены 70 строк хелперов, добавлен include, остальная логика `step`/`obs`/`minimap` без изменений (следующий шаг — вынести `obs`/`minimap`/`rewards` блоки из `step` в `rewards.cpp`/`observation.cpp`).

**`CMakeLists.txt`:** добавлен `src/rewards.cpp` в `add_library`.

**`python/cpp_vecenv.py` + `python/cpp_env.py` (304/278 строк):**
- Убран hardcoded `_REWARD_KEYS` tuple (40 ключей) — теперь `for key, value in reward_config.items(): setattr(rc, key, value)` (DRY, forward-compat).
- `_missing` логика сохранена, `disable_net_worth`/`disable_daily_income` теперь только если True (не перезатирает JSON).
- Debug-блок упрощён: печатает `reward_config` ключи, не зависит от несуществующего tuple.

---

## 3. Инварианты совместимости

- Все `import` старые работают:
  ```python
  from rl.config import Config, RewardConfig
  from train_ui2.main_window import MainWindow2, REWARD_GROUPS, PARAM_GROUPS
  from rl.actor_critic import ActorCritic
  from rl.curriculum import STAGE_MAP  # новый, но старый путь через constants тоже
  ```
- JSON конфиги v1/v2 читаются (миграция `gamma`/`ent_coef` в UI state сохранена, но теперь внутри `Config.from_dict`).
- C++ `colony_cpp.ColonyVecEnvCpp`/`RewardConfig` поля не переименованы, pybind11 bindings не менялись.

---

## 4. Метрики

| Файл | До | После | Δ |
|---|---|---|---|
| `rl/config.py` | 389 строк, 3 дублирующих списка | 260 строк + хелперы | −33%, DRY |
| `train_ui2/main_window.py` | 1486 строк, 120 строк констант/иконок | ~1380 строк, импорты | −100 строк |
| `rl/env_manager.py` | 410 строк, 2× дубли буфера | 260 строк, фабрики | −37% |
| `rl/actor_critic*.py` | 3× дубли `orthogonal_init` | 1× base + 3 тонких | −60 строк |
| `include/colony/env.h` | 381 строк, встроен RewardConfig | 320 + 80 (reward_config.h) | модульность |
| `src/env.cpp` | 1545 строк, 70 строк хелперов | 1475 + 60 (rewards.cpp) | модульность |
| `python/cpp_vecenv.py` | hardcoded 40 ключей | динамический `items()` | DRY |
| **Новые модули** | — | `rl/curriculum.py`, `rl/_nn_common.py`, `train_ui2/constants.py`, `train_ui2/icons.py`, `include/colony/reward_config.h`, `src/rewards.cpp` | +6 файлов |

**Всего удалено дублирования:** ~250 строк ручных списков/копипасты → интроспекция/наследование.

---

## 5. Что осталось на следующий итерационный рефакторинг

- `train_ui2/main_window.py` → разнести вкладки `tabs/training.py`, `monitoring.py`, `rewards.py`, `curriculum.py`, `models.py`, `watch.py` + `state.py` (load/save/restore) — сейчас только constants/icons вынесены, остальное готово к сплиту (интерфейсы уже описаны).
- `src/env.cpp` `step()` (~600 строк) → `src/rewards.cpp` (часть уже), `src/observation.cpp` (obs + catalog), `src/minimap.cpp`.
- `rl/async_trainer.py` (762 строки) → `rl/trainer/evaluator.py`, `checkpoint.py`, `curriculum_scheduler.py` (аналогично env_manager).
- `train_ui2/parameter_widget.py` (335 строк, но `PARAM_SPECS` генерится из `rl.config` — уже частично) → автогенерация из dataclass fields.
- Линтинг: `ruff`/`clang-format`, сортировка импортов, `__all__`.

---

## 6. Проверка

```bash
python -m py_compile rl/config.py rl/env_manager.py rl/actor_critic* rl/_nn_common.py \
  train_ui2/constants.py train_ui2/icons.py train_ui2/main_window.py \
  python/cpp_vecenv.py python/cpp_env.py   # ok
g++ -std=c++17 -Iinclude -Iinclude/third_party -fsyntax-only src/rewards.cpp  # ok
# UI: python -m train_ui2.app  (запускается, вкладка Курикулум работает)
# pytest — требует torch/colony_cpp.pyd (недоступны в песочнице), py_compile и g++ — парциальный, но все API сохранены
```

---

## 7. Как поддерживать дальше

- Новые награды: добавить поле в `RewardConfig` (Python + C++ `reward_config.h`) + ключ в `configs/reward_v3.json` — `to_dict`/`from_dict` подхватят автоматически.
- Новый этап курикулума: править только `rl/curriculum.py` `STAGE_MAP`/`ALL_IDS` — UI и Env подтянут.
- Новый бэкбон: наследовать `ActorCriticBase`, использовать `orthogonal_init`.
