# Curriculum: «поставил только водоканал — а он строит прииск»

`docs/CURRICULUM_FIX_2026_09.md`

## 1. Симптом

Вкладка **Курикулум**: этап `0`, чекбокс «Ручной набор из Курикулума» включён,
отмечен **только Водоканал**. Ожидается: агент может строить только Водоканал.
Фактически: строит прииск (Goldmine) и другие здания — ручной набор теряется.

## 2. Причина (найдена в C++, `src/env.cpp`)

Конструктор среды ручной набор **учитывал** (`if (!unlock_ids.empty()) { … has_unlocked_ = true; }`),
поэтому в момент создания среды всё было правильно. Но **`set_curriculum_stage()`
его стирал**:

```cpp
void ColonyEnvCpp::set_curriculum_stage(int stage) {
    curriculum_stage_ = stage;
    unlocked_.clear();          // ← ручной набор удалён
    has_unlocked_ = false;      // ← «ограничений нет»
    if (stage > 0) { … только пресет этапа … }
    compute_catalog();
}
```

`has_unlocked_ == false` означает «разрешено всё» — в маске действий
(`env.cpp`, guard по `has_unlocked_`) сразу открываются все 32 постройки,
включая прииск. То есть сценарий «этап 0 + ручной набор» работал ровно до
первого переприменения этапа, после чего молча превращался в «этап 0 без
ограничений». Переприменений в коде несколько:

* расписание этапов (`--curriculum-schedule`, `AsyncTrainer`);
* команда `reset_curriculum` («сбросить курикулум») из UI;
* окно «Наблюдение»: `watch_champion.py` применяло этап к среде;
* `ColonyVecEnvCpp::set_curriculum_stage()` раздавал этап всем под-средам,
  нигде не возвращая ручной набор.

Дополнительно ручной набор вообще не доходил до сред, создаваемых в обход
обучения: `train_ui2/evaluator.run_eval()` (eval/отбор лучшей модели) и
`watch_champion.py` (наблюдение) создавали `CppColonyEnv` без `unlock_ids`,
а GUI-экзешник — вообще без доступа к набору из UI.

## 3. Что исправлено

**Единая точка правды — `rl/curriculum.py`** (`allowed_ids(stage, unlock_ids, use_curriculum_tab)`,
`manual_ids_csv`) плюс «жёсткое» состояние в C++:

| Файл | Изменение |
|---|---|
| `src/env.cpp` | `rebuild_unlocked()` = пресет этапа ∪ `manual_unlock_ids_`; `set_curriculum_stage()` больше **не стирает** ручной набор (и на этапе 0 он остаётся единственным разрешённым списком); новый `set_unlock_ids()` |
| `include/colony/env.h` | поле `manual_unlock_ids_`, `rebuild_unlocked()`, `unlock_ids()`; `ColonyVecEnvCpp::set_unlock_ids/unlock_ids` раздаёт набор всем под-средам |
| `src/bindings.cpp` | `set_unlock_ids` / `unlock_ids` в Python-биндингах обеих сред |
| `rl/env_manager.py` | `_make_vec_env` берёт `cfg.effective_unlock_ids()`; `set_curriculum_stage()` повторно применяет ручной набор (и предупреждает, если .pyd старый) |
| `rl/config.py` | `effective_unlock_ids()` / `curriculum_meta()` — учёт чекбокса в одном месте |
| `rl/async_trainer.py` | этап **и** `unlock_ids` **и** чекбокс пишутся в `meta.json`/`best_model.meta.json`; eval вызывается с тем же курикулумом; после турнира мета обновляется |
| `train_ui2/evaluator.py` | `run_eval(..., curriculum_stage, unlock_ids, use_curriculum_tab)`: приоритет — аргументы, иначе значения из меты модели; в лог пишется `[Eval] curriculum: …` |
| `watch_champion.py` | курикулум берётся из CLI → меты модели, среда создаётся сразу с ним; BUILD-действия маскируются по разрешённому списку; `--unlock-ids` |
| `src/gui.cpp` | новые ключи `--unlock-id` (повторяемый) и `--unlock-ids` (CSV) — визуальное окно строится с тем же набором |
| `train.py` | `--curriculum-stage`, `--unlock-ids`, `--use-curriculum-tab`; в консоль печатается итоговый список |
| `train_ui2/worker.py` | в лог UI пишется `curriculum: stage=… manual=… checkbox=… → N buildings allowed` |

Ручной набор по-прежнему действует только при включённом чекбоксе (выключен —
работает чистый пресет этапа), а на этапах 1–3 набор объединяется с пресетом.

## 4. Проверка

**C++ (сценарий из отчёта), `tests/cpp/curriculum_check.cpp`** — сборка и запуск
из корня репозитория (команды в шапке файла):

```
unrestricted env (5M cash): 14/32 build actions available
[ok]  manual-only env allows exactly the selected id (no other building)
[ok]  прииск (Goldmine) is masked
[ok]  set_curriculum_stage(0) keeps the manual-only set
[ok]  env still stores the manual unlock_ids after the stage switch
[ok]  stage 2 keeps the manual id … stage 2 -> 0 restores the manual-only set
[ok]  locked Goldmine action is refused by the env (nothing built)
ALL CHECKS PASSED (0 failure(s))
```

Те же проверки, собранные с **прежней** версией `src/env.cpp`
(коммит `847e5d5`, файл можно достать через `git show 847e5d5:src/env.cpp`),
дают `BUG REPRODUCED: stage switch dropped the manual set` —
то есть сначала баг воспроизведён, потом исправлен.

**Python**: `python -m pytest tests/test_curriculum_scope.py -q` → `19 passed, 2 skipped`
(в т.ч. eval восстанавливает набор из `best_model.meta.json`, аргументы
перекрывают мету, `resolve_curriculum` не путает «не задано» и «пусто»).
Полный прогон `pytest -q` не изменил набор падающих тестов относительно
`main` (падают только те, которым нужен `colony_cpp`).

## 5. Что нужно сделать локально

1. **Пересобрать расширение**: `build_pyext.bat` → `python/colony_cpp.pyd`
   (в нём `set_unlock_ids` и неразрушающий `set_curriculum_stage`).
2. **Пересобрать GUI**: `build_gui.bat` → `sakhalin_colony_gui.exe`
   (визуальному окну нужен `--unlock-ids`).
3. Запустить обучение и убедиться по логу в UI:
   `[Worker] curriculum: stage=0 manual=WaterChannel checkbox=True → 1 buildings allowed`
   (в eval — `[Eval] curriculum: stage=0, manual=WaterChannel, allowed=1 buildings`,
   в наблюдении — `Curriculum … → 1 buildings allowed` и `Action mask: N locked build action(s) forced off`).
