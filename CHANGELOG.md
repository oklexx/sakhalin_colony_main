# Changelog

Формат: `## [дата]` + список изменений. Завершённые задачи из [STATE.md](STATE.md)
переносятся сюда. Источники: `docs/*_2026_09.md`, аудиты, README §11.

## [2026-09-19]

### Исправлено
- **Синхронизация дефолтов наград с каноном (золотое правило RULES.md)**:
  C++ `RewardConfig` (include/colony/reward_config.h) отставал от
  `configs/reward_v4.json`/`rl/config.py` и оставался на v3-значениях —
  11 полей приведены к v4: `build_bonus` 2.0→1.2, `build_cost_penalty`
  0.0001→0.00004, `loan_penalty` 0.5→2.0, `novelty` 5.0→3.0,
  `survival_coeff` 0→0.0005, `milestone_base_bonus` 30→10, клип ±50→±100,
  `death_penalty` 20→12, `born_bonus` 1→2, `debt_coeff` 0.1→0.
  Консольные/GUI-сборки и сырые биндинги теперь считают в каноническом v4,
  как и RL-путь (который всегда переопределял поля явно).
- **Дефолт obs — v2 (299) на всех границах**: `Curriculum::obs_version` и
  `Curriculum::from_json` теперь по умолчанию 2 (было 0 — сырой C++ env
  молча отдавал 248-мерный obs). rl/curriculum и python-обёртки и так
  передавали 2 явно; расхождение было классом «тихий рассинхрон границ».

### Тесты
- Полный `pytest` зелёный: **318 passed, 2 skipped** (было 19 failed + INTERNALERROR
  от `test_smoke_fixes.py`). Прогон впервые выполнен под Linux (сборка
  colony_cpp.so через CMake + pybind11; см. STATE.md).
- **Добавлен `tests/test_review_plan_regressions.py`** — регрессии Этапа 0
  плана ревью: `delete_base` не портит spatial-индекс (нет фантомов «занято»),
  единый валидатор `Game::build` (не та земля / сгоревший участок / занятая
  клетка), market buy/sell отвергает отрицательные количества, размер Sunduk
  в биндингах, `ep_return` == сумма возвращённых reward, VecEnv fail-fast
  (n_envs=0, размеры seeds/actions), python-обёртка не затирает
  `terminal_observation` из C++-info.
- Обновлены устаревшие тесты, отставшие от легитимных изменений:
  - профиль наград v3→v4 (`test_reward_default_profile`, `test_smoke_fixes`,
    `test_build_cost`, `test_milestones`, `test_reward_clip`,
    `test_colony_robustness` — 42→47 ключа);
  - 45→49 действий (ROAD_E/W/S/N): `test_async_trainer`, `test_evaluator`
    (там же починен неэффективный monkeypatch `CppColonyEnv`);
  - дефолт obs v1→v2: `test_curriculum_contract`, `test_curriculum_scope`
    (включая счётчик call-site'ов передачи курикулума в run_eval — поведение
    верное, проверка источника стала точнее).
- C++-пробы (tests/cpp) выровнены с каноном: `reward_regressions` R5 теперь
  берёт направление на воду с карты (хвост obs v2 — направления к дереву/углю/
  …/золоту, не вода), `p0_p1_check` проверяет механизм клипа на явных границах,
  `road_direction_check` D4 — greedy к воде с карты, `curriculum_check` —
  избирательность гейта проверяется на Road, а WaterChannel на суше ожидаемо
  отвергается валидатором размещения (не гейтом). Все пробы — 0 fail.

## [2026-09-17]

### Добавлено
- **Профиль наград `reward_v4.json`** — канонический: терминальный бонус за
  выживание (`goal_survival_coeff=200`), бонус за чистую оплату главного налога
  (`main_tax_cash_bonus=100`), предналоговое давление
  (`main_tax_pressure_coeff=0.002`), `debt_coeff=0` (отмена двойного процента
  долга), дисциплина кредитов (`loan_penalty=2.0`), балансировка стройки
  (`build_bonus=1.2`, `build_cost_penalty=0.00004`), клип наград [−100, +100].
  Обоснование — `docs/AUDIT_REWARD_2026_09.md`.
- **P1: маски менеджеров по применимости** (`mask_managers_by_applicability`):
  `SELL_SURPLUS` — только при излишке, `BUY_FOOD` — только при нужде,
  `REPAY_LOAN` — только при долге, `IMPROVE_LAND` — только если хватает на
  участок; `manual_tax` — никогда.
- **P0: налоговый долг (`tax_to_debt=true`)** — в RL-среде неоплаченный налог
  автоматически уходит в кредит, календарь не замирает (убран «налоговый клин
  365-го дня»).
- **obs v2 (дефолт, 299 чисел)**: `(dx, dy)` к ближайшему тайлу дерева/угля/
  железа/нефти/золота; v0/v1 — строгие префиксы.
- **Mechanic curriculum gate**: allow-list механик
  (`improve_land`/`preservation`/`credit`) с расписанием разблокировки
  (`docs/MECHANIC_CURRICULUM_2026_09.md`).

### Исправлено
- R4/R5 (аудит наград): см. `docs/AUDIT_REWARD_2026_09.md`.
- B1 (дефолты `RewardConfig.from_dict`), B2 (eval без нормализации), B6/B7
  (approx_kl ×12, no-op LR-scheduler), N1 (знак manual_tax_penalty), D1
  (~1 ГБ GPU→CPU копий), F3/N7 (score на 76% из days), N3–N5 (хардкод-награды).

## [2026-09-16]

- Ревью режимов наблюдения `flat`/`minimap`/`hybrid`
  (`docs/REVIEW_WATER_MODES_2026_09.md`): режимы корректны; водоканал не
  строится из-за маски действий и road-spam-аргмакса (зафиксировано в STATE.md
  как главная открытая проблема).

## [2026-09-11]

- **Полный совместимый рефакторинг** (`docs/REFACTOR_2026_09.md`): `rl/config.py`
  на интроспекции dataclass, монолиты → модули 200–400 строк, единый источник
  правды наград/курикулума, типизация и docstrings.
- **Аудит 10 пунктов** (`docs/AUDIT_2026_09.md`): паритет
  `food/water_need_bonus 2.0→0.8`, вкладка «Курикулум» в UI, 7-й блок наград,
  `need_fill_bonus` двухступенчатый, защита от NaN, чистка репозитория.
- Диагностика стабильности PPO (`docs/RL_DIAGNOSIS_2026_09.md`): шесть
  измеренных причин («модель не строит первую цепочку»), P0/P1 — выполнены.
- Аудит системы наград (`docs/AUDIT_REWARD_2026_09.md`): зонды 7 политик,
  ручной план-эталон, регрессии R1–R4.

## [ранее]

- A1 (`avg_return`), A2 (`is_hybrid`), A3 (`load_from_file`), B5
  (`_last_terminateds`) — исправлены и проверены.
- Нормализация obs/reward (RunningMeanStd) в C++, детерминированные сиды
  (base + i×10000), GAE с маской бутстрапа, 45 действий консистентны во всех
  точках (обучение/eval/watch).
- GUI-дашборд `train_ui2/` (PySide6), `watch_champion.py`, бенчмарк
  `tests/bench_per_step.py`.
