# Код-ревью (2026-09-24): правки последней недели + старые незамеченные баги

Область: всё, что менялось 2026-09-16…09-23 по CHANGELOG.md (`rl/async_trainer.py`,
`rl/action_monitor.py`, `rl/rollout_buffer.py`, `rl/env_manager.py`, `rl/curriculum.py`,
`rl/config.py`, `train_ui2/*`, `python/cpp_vecenv*.py`, `src/vec_env.cpp`, `src/gui.cpp`,
`watch_champion.py`), плюс соседний код, который эти правки затронули.

Метод: статическое чтение + прогон того, что запускается в этом окружении
(ruff, pytest, torch-репро на CPU). `colony_cpp` в песочнице не собран (нет `Python.h`),
`libGL.so.1` отсутствует → Qt-тесты и любой реальный прогон среды не выполнялись;
все выводы по C++/среде — статические. Это отмечено в каждом пункте.

Сводка: **2 P0**, **4 P1** (+1 найден на Этапе 1: P1-5), **3 P2**, **~8 P3/техдолг**. P0-1 и P0-2 — регрессии
последней недели (тихие: тесты их не ловят).

---

## 1. Баги

### P0-1. `watch_champion.py`: headless-режим падает с `UnboundLocalError` (`watch_temp`)

*Файл:* `watch_champion.py:1181-1182` (использование) и `1231` (присваивание), внутри `main()` (837-1294).

`watch_temp` — локальная переменная `main()`. Первое использование стоит в
основном (headless) цикле прогона эпизодов:

```python
if not args.visual:                     # 1138 — режим ПО УМОЛЧАНИЮ (--visual = store_true)
    for ep in range(args.episodes):
        ...
        if watch_temp > 0.0:            # 1181  <-- до присваивания
            sample_probs = torch.softmax(logits / max(1e-4, watch_temp), dim=-1)...
```

а присваивание `watch_temp = args.temperature if ... else (0.8 if args.sample else 0.0)`
находится **ниже**, уже после блока headless (1231), и достигается только на visual-пути (1280).

Проверка:
* поиск по файлу: `watch_temp` встречается ровно 4 раза — 1181, 1182 (чтение), 1231 (запись), 1280 (visual);
* `ruff check --select F821` → `watch_champion.py:1181/1182: Undefined name 'watch_temp'`;
* семантика Python: любая функция, где имя присваивается ниже по тексту, считает его локальным →
  `UnboundLocalError: cannot access local variable 'watch_temp'`.

Итог: `python watch_champion.py --model-dir <run>` (команда из README, строка 55) и любой
прогон без `--visual` падают на первом же шаге выбора действия. `--sample`/`--temperature`
на headless-пути не только не работают — они вообще недостижимы.

Почему не поймали: `tests/test_watch_champion.py` проверяет `--help` (subprocess) и юнит-функции
`launch_visual_watch`/`write_action_file`/`read_state_file`, но не прогон `main()`;
`tests/test_watch_visual.py` всегда запускает с `--visual`
(`argv = ["watch_champion.py", "--model-dir", ..., "--visual", ...]`), а существующий
статический тест `tests/test_trainer_bootstrap.py::test_module_global_names_resolve`
проверяет только **глобальные** имена, не «локальное до присваивания».

**Фикс:** перенести вычисление `watch_temp` выше `if not args.visual:` (строка ~1137),
до цикла; заодно вынести в одну переменную `watch_temp = ...` рядом с остальными
производными от args. **Регресс:** subprocess-тест headless-прогона
(`--episodes 1 --max-steps 2` с собранным `colony_cpp`) + расширение
`test_module_global_names_resolve` на symtable-проверку «локальное имя читается до присваивания».

---

### P0-2. `_TensorRolloutBuffer.add` теряет `trunc_value` → GAE-бутстрап на усечении не работает в minimap/hybrid

*Файлы:* `rl/rollout_buffer.py:228` (сигнатура подкласса принимает `trunc_value`),
`rl/rollout_buffer.py:235` (вызов суперкласса без него), `rl/env_manager.py:318,335,348`.

Подкласс объявляет параметр и **не пробрасывает** его в базовый `add`
(`rl/rollout_buffer.py:59-71`, где `trunc_value` есть и обрабатывается):

```python
def add(self, obs, action, reward, log_prob, value, done, terminated=None,
        flat=None, action_masks=None, trunc_value=None):     # 228
    super().add(obs, action, reward, log_prob, value, done, terminated, action_masks)  # 235
```

`env_manager` в minimap/hybrid-режимах (`obs_mode != "flat"`) создаёт именно
`_TensorRolloutBuffer` (`rl/env_manager.py:95-109`) и передаёт `trunc_value=trunc_value_t`
(строки 335 и 348). В результате для tensor-буфера `trunc_values` всегда остаётся нулевым,
и `compute_gae` на усечении (`done & ~terminated`) подставляет ноль вместо `V(s_T)` —
т.е. ровно та ошибка, которую правка 2026-09-22 («GAE truncation bootstrap») устраняла для flat-буфера.

Репро (CPU, torch 2.14):

```python
flat = RolloutBuffer(n_steps=4, n_envs=1, obs_size=3, n_actions=4, gamma=0.99, gae_lambda=0.95, device=torch.device("cpu"))
tens = _TensorRolloutBuffer(n_steps=4, n_envs=1, obs_shape=(8,32,32), n_actions=4, gamma=0.99, gae_lambda=0.95, device=torch.device("cpu"))
# шаг 1: done=[1], terminated=[0], trunc_value=[7.0]; остальные шаги — нули
# результат:
flat  trunc_values: [0.0, 7.0, 0.0, 0.0]
tensor trunc_values: [0.0, 0.0, 0.0, 0.0]     # <-- потеряно
```

Почему не поймали: `tests/test_gae.py` импортирует и проверяет **только**
`from rl.rollout_buffer import RolloutBuffer` (строка 9) — ни одного теста на `_TensorRolloutBuffer`.

**Влияние:** все прогоны minimap/hybrid (в т.ч. водяной A/B, `docs/WATER_HYBRID_AB_2026_09.md`)
считают value-таргеты на границах эпизодов неверно; сравнение flat- и hybrid-вариантов,
сделанное до фикса, недостоверно.

**Фикс:** `super().add(obs, action, reward, log_prob, value, done, terminated, action_masks, trunc_value)`.
**Регресс:** параметризовать существующие тесты `tests/test_gae.py` по обоим классам
буфера (flat + tensor) и/или добавить проверку `buffer.trunc_values` после `add`.

---

### P1-1. UI «Стоп» убивает воркер жёстко: `final_model.pt`, `meta.json` и `run_end` теряются

`train_ui2/main_window.py:1174-1183`: сначала в файл команд пишется `stop_training`, и сразу же
вызывается `self._train_proc.terminate()` (SIGTERM / `TerminateProcess`). Graceful-путь в трейнере
есть — `final_model.pt` + `.norm.json` (`rl/async_trainer.py:1073`), финальный турнир за
`best_model.pt`, `_episode_diagnostics.close(status="stopped")` (строка 1235), — но до него дело не
доходит: процесс умирает мгновенно, а очередь команд трейнер читает только между роллаутами
(`_process_commands` вызывается на итерации обучения, `rl/async_trainer.py:832`).

Последствия остановки через UI:
* нет `final_model.pt`/`.norm.json` и `meta.json` (`train_ui2/worker.py:390-404` пишет его уже после `train()`);
* `episode_diagnostics.jsonl` остаётся без записи `run_end` — файл неотличим от «прогон упал»;
* финальный турнир не запускается, `best_model.pt` может не соответствовать лучшему чекпойнту.

**Фикс:** после записи команды не убивать сразу: дождаться завершения (индикатор «Останавливаем…»,
таймаут 30-60 с), `terminate()` — только fallback; в трейнере опрашивать команды чаще (например,
раз в N шагов внутри `_collect_rollout`, рядом с `_check_stop`), чтобы стоп срабатывал за секунды.
При остановке по команде сохранять `final_model.pt`/`.norm.json`/`meta.json` и писать `run_end`
(со `status="stopped"`), а долгий финальный турнир либо пропускать, либо отмечать в мете как
«не выполнялся» — иначе «мягкий стоп» будет означать ожидание в минуты. Дополнительно: закрывать
диагностику по сигналу (`signal.signal(SIGTERM, ...)` + `atexit`), чтобы `run_end` появлялся и при
жёстком убийстве процесса.
**Тест:** послать `stop_training` → появились `final_model.pt`, `meta.json` и `run_end` со `status="stopped"`.

---

### P1-2. `train_ui2/worker.py:161`: `except (json.JSONDecodeError, _queue.Full, Exception): pass`

Конструкция ловит **всё** (третий элемент кортежа), поэтому:
запись команды в очередь (`command_queue.put`, `timeout=1`) при заполненной очереди
или при разборе битой строки **молча теряется** — кнопки «Пауза/Стоп/Энтропия» могут
не сработать без единого сообщения в лог/UI. Это прямое нарушение RULES.md
(«без тихих `except: pass`») и ровно тот класс, который в 09-20 вычищали из трейнера.

**Фикс:** разнести случаи — `json.JSONDecodeError` (предупреждение в лог/msg-файл),
`_queue.Full` (повтор с задержкой или явная ошибка в UI), `Exception` — не ловить вообще.
**Тест:** юнит-тест на «битая строка команды → предупреждение, валидная после неё обрабатывается».

---

### P1-3. Резюме чекпойнтов: `train.py:256` (strict=True) vs `train_ui2/worker.py:334` (strict=False)

`rl/actor_critic.py:42` документирует контракт «старые чекпойнты грузятся с `strict=False`»
(после 2026-09-22 добавлены `actor_mask_proj`/`critic_mask_proj`), `rl/ppo.py:389` его соблюдает,
UI-файнтюн (`worker.py:334`) — тоже. А CLI-резюм в `train.py:256` вызывает
`em.model.load_state_dict(clean)` без `strict=False` → любой flat-чекпойнт старше 09-22
падает с `RuntimeError: Missing key(s) ... actor_mask_proj.weight, critic_mask_proj.weight`,
хотя загрузка безопасна (проекции инициализированы нулями → поведение старой модели сохраняется).

**Фикс:** общий хелпер `load_policy_state(model, state, ckpt_path)` в `rl/_nn_common.py`:
имена ключей чистятся от `_orig_mod.`, загрузка `strict=False`, отчёт о missing/unexpected
в лог; использовать в `train.py`, `train_ui2/worker.py`, `rl/ppo.py`, `watch_champion.py`.
**Тест:** чекпойнт без `*_mask_proj` грузится, проекции остаются нулевыми, `ValueError` для чужого obs-размера.

---

### P1-4. Протокол команд: два словаря, мёртвая ветка и тихое игнорирование неизвестных команд

В `train_ui2/worker.py` живут **два** читателя команд с разными словарями:

* `_watch_stdin` (~строки 90-125) понимает `pause | resume | boost_entropy | reset_curriculum | stop`
  (это ровно значения `train_ui2/protocol.py:182-188` `CommandType`) — но UI в stdin **не пишет**
  (`train_ui2/main_window.py` не использует stdin вовсе), т.е. путь практически мёртв;
  при этом `pause`/`resume` из него кладутся в очередь как есть, а `AsyncTrainer._process_commands`
  (`rl/async_trainer.py:475-508`) знает только `pause_training`/`resume_training` → **тихий no-op**;
* `_watch_commands_file` (строки ~128-163) читает файл, куда UI пишет `stop_training` /
  `pause_training` / `resume_training` / `boost_entropy` (`main_window.py:1178,1189,1194,1202`) —
  этот путь рабочий (строки совпадают с трейнером).

Дополнительно: в `_process_commands` нет ветки `else`, поэтому любая незнакомая команда
(опечатка, старый формат, `start`) исчезает без записи в лог. Итого контракт существует в
трёх несогласованных видах, а ошибка в нём не диагностируется.

**Фикс:** один источник строк — константы в `protocol.py` (UI, воркер, трейнер берут их оттуда),
enum привести к реальным значениям (или удалить как мёртвый), в `_process_commands` добавить
`else: self._log(f"[Command] unknown: {cmd}")`. **Тест:** паритет
«множество команд, которые отправляет UI ⊆ множество, которое обрабатывает трейнер» +
неизвестная команда пишет предупреждение.

---

### P2-1. `train_ui2/main_window.py:748, 764`: голые `except: pass`

`_sort_curriculum_table` и `_collect_curriculum_schedule` глушат **любую** ошибку при
разборе таблицы расписания: строка с опечаткой просто исчезает из расписания обучения,
и пользователь видит «курикулум как будто не применился» (ровно тот класс, что описан
в `docs/CURRICULUM_FIX_2026_09.md`). Также ловится `KeyboardInterrupt`/`SystemExit`.

**Фикс:** `except (AttributeError, ValueError)` + сообщение в статус-бар («строка N пропущена»);
вынести разбор строки в чистую функцию и покрыть тестом.

---

### P2-2. `python/cpp_vecenv.py:289-291`: `get_attr` игнорирует `indices`

```python
def get_attr(self, attr_name, indices=None):
    target_envs = [self] * self.num_envs if indices is None else [self]
    return [getattr(env, attr_name) for env in target_envs]
```

При любом непустом `indices` возвращается список **длины 1** вместо `len(indices)`
(и теряется привязка к индекс-порядку). Сейчас метод никем не вызывается, но это часть
VecEnv-API (SB3-совместимость) — латентная ловушка для будущих обёрток/мониторинга.

**Фикс:** `[self] * len(list(indices))` (или явный `NotImplementedError` с сообщением),
тест на длину и порядок.

---

### P2-3. `python/cpp_vecenv.py:230`: `terminal_observation` подменяется obs нового эпизода

```python
if dones[i]:
    if "terminal_observation" in info: ...      # C++ отдаёт RAW s_T (src/vec_env.cpp:192-193)
    else:
        info["terminal_observation"] = obs[i].copy()   # <-- это obs ПОСЛЕ auto-reset
```

В `step_wait` массив `obs` уже содержит первую обсервацию **нового** эпизода (auto-reset),
поэтому «запасной» вариант кладёт в `terminal_observation` совсем не терминальное состояние.
Сейчас это почти мёртвая ветка (C++ 09-22 всегда пишет поле), но при старом `.pyd`
(`COLONY_ALLOW_STALE_PYD`) или в новом режиме она «спасёт» вызов, отдав неверный `V(s_T)`
в `env_manager._truncation_bootstrap_values` (`rl/env_manager.py:394-458`) — тихо и без следа.

**Фикс:** не выдумывать значение: оставить ключ отсутствующим и один раз предупредить
(`_monitor_warn_once`-подобно) — тогда truncation-bootstrap честно не считается, а не считается неверно.

---

### P3-1. `rl/config.py:437`: `-> CurriculumState` без импорта (ruff F821)

Аннотация ссылается на имя, которого нет в модуле (`from __future__ import annotations`
спасает runtime, но `typing.get_type_hints()`/mypy падают). Фикс: `TYPE_CHECKING`-импорт
из `rl.curriculum`.

### P3-2. `tests/test_reward_clip.py:117`: NameError в сообщении ассерта

`f"Reward {r} outside clip range [-2, 2]"` — переменной `r` нет: при реальном нарушении
клипа тест упадёт с `NameError`, а не с диагностикой (и будет выглядеть как инфраструктурная
поломка). Фикс: `{reward}`.

### P3-3. Гигиена: ruff-долг 739 замечаний, шага lint в CI нет

`ruff check .`: **739** замечаний. Большинство — не баги, а модернизация типов:
`UP006`×233, `UP045`×111, `UP035`×53, `UP037`×10, `UP015`×5 (это 412 из 739 и
чинится одним `ruff check --fix`). Реально значимые группы:
`I001`×105 (порядок импортов), `E702`×49 (несколько выражений через `;`),
`F401`×45 (неиспользуемые импорты), `W293`×39 (пробелы в пустых строках),
`E402`×25 (импорт не в начале модуля), `B007`×15, `F841`×13,
`B905`×6, `E741`×5, `F821`×4, `F541`×4, `F811`×3, `E722`×3.
`pyproject.toml` уже настраивает ruff/mypy, а `.github/workflows/ci.yml` их не запускает.
Конкретное, что стоит починить в первую очередь:
* `train_ui2/parameter_widget.py:48 и 383` — `spec_for` определён дважды (второе определение мёртвое, F811);
* `rl/async_trainer.py:844,860` — `rollout_time` и локальный `loop_detected` не читаются
  (F841; метрика строится из `envs_with_loops` — код детектора циклов недоделан, ср. STATE.md);
* `B905` (`zip()` без `strict=`): `rl/async_trainer.py:894`, `train_ui2/worker.py:295`;
* `E402` — проверить, что это только скрипты/тесты (в модулях `rl/` такого быть не должно);
* `E722` бар-эксепты — см. P2-1; `F821` — см. P3-1 и P0-1.

### P3-4. `src/gui.cpp:448-472`: невалидное действие не «поглощается» → возможен зависание watch

`ai_read_action()` удаляет файл только тогда, когда значение распарсилось и `action >= 0`.
Если в файле окажется `-1` (отладка, чужой процесс, остаток), окно будет возвращать `-1`
на каждом кадре, а драйвер — вечно ждать `state.json`: тот же сценарий «тихого зависания»,
который чинили 2026-09-20. Фикс: считать файл потреблённым и при отрицательном значении
(с записью в debug-лог), либо явно сообщать об ошибке в `state.json`.

### P3-5. `rl/episode_diagnostics.py:26-41`: NaN/Inf проходят в `json.dumps(..., allow_nan=False)`

`_json_safe` пропускает `float` как есть; нефинитный reward/day (бывает при расходящемся
обучении) даст `ValueError` внутри `append_episode` — а он вызывается из трейнера под
`except Exception` с «warning once», т.е. диагностика тихо прекратится. Фикс: заменять
нефинитные числа на `null` (или строку) в `_json_safe`.

### P3-6. `src/vec_env.cpp:65-66, 182-187`: `terminal_minimap` без предупреждения

Буфер терминальных миникарт выделяется всегда (`n_envs × 8×32×32` float — ~32 КБ на env
даже в flat-режиме), а если `env.minimap().size() != 8*32*32`, поле молча не заполняется
(`terminal_minimap_valid_[i] = 0`), и `_truncation_bootstrap_values` тихо пропускает
бутстрап (нулевой вместо `V(s_T)`). Стоит: (а) не считать миникарту, когда режим её не
использует; (б) при несовпадении размера — один раз предупредить в info (`"terminal_minimap_missing": true`),
чтобы это было видно в диагностике, а не выглядело как «все эпизоды настоящие терминальные».

### P3-7. `rl/action_monitor.py`: «легальность» — это доля *доступности*, а не легальность выбранного действия

`ActionLegalityMonitor.add_step` суммирует **маски** (`mask==1`) по шагам и env, т.е. считает
долю шагов, на которых действие было *доступно*, а не долю шагов, где выбранное действие
было легальным. Для задачи «отличить „нельзя“ от „не хочет“» это работает, но подпись
в UI («легально») и докстринг вводят в заблуждение — стоит переименовать в «доступно, %»
или считать фактическую легальность выбранных действий (второй вариант — одно сложение
с маской по индексу действия).

---

### P3-8. Мёртвый код рядом с правками последней недели

* `rl/async_trainer.py:1249` `_get_steps_in_curriculum_stage()` — не вызывается ниоткуда
  (остаток старого расчёта прогресса этапа; заменён на `rl.curriculum.stage_progress`);
* `train_ui2/worker.py:49` `--command-queue-size` — аргумент CLI читается в `args`, но нигде не
  используется (`queue.Queue()` создаётся без `maxsize`);
* `rl/async_trainer.py:844,860` — `rollout_time`, локальный `loop_detected` (см. P3-3).

Либо удалить, либо подключить; иначе следующий рефакторинг будет опираться на «работающий» код.

---

## 2. Что сделано хорошо (чтобы не сломать при исправлениях)

* Flat-путь GAE с усечением (`rl/rollout_buffer.py:103-157`) реализован корректно и покрыт тестами;
  `terminated` и `dones` разделены по всей цепочке (C++ → `CppVecEnv._last_terminateds` → буфер → GAE).
* Атомарный IPC-обмен GUI↔драйвер (`src/gui.cpp:474-535`, per-run каталог, `tmp+rename`,
  fallback на `copy_file` для Windows) — с внятными комментариями о причинах.
* Единый источник правды по курикулуму (`rl/curriculum.py` + `build_state`/`resolve_state`),
  fail-closed разбор, `mechanics_enabled_at_step` с запретом «повторного закрытия» механик.
* Мониторные метрики (09-23) аккуратно изолированы: любая ошибка наблюдательного счётчика
  не валит обучение и **громко** пишется один раз (`_monitor_warn_once`).

---

## 3. План исправлений

### Этап 0 — «красное» (0.5-1 день): регрессии, ломающие пользовательские сценарии — ✅ СДЕЛАНО 2026-09-24
- [x] **P0-1**: вычисление `watch_temp` (и предупреждение о `--map-size`) поднято выше
      headless-цикла; реальный `main()` без `--visual` прогоняется тестом
      `tests/test_watch_visual.py::test_headless_watch_runs_episode` (argmax + `--sample`).
- [x] **P0-2**: `trunc_value` пробрасывается в `super().add(...)`; все GAE-тесты
      параметризованы по flat/tensor, добавлены `test_tensor_buffer_stores_trunc_value`
      (+ hybrid `flat_dim>0`) и `test_buffer_kinds_agree_on_gae`.
- [x] Регресс-страховки:
      (а) smoke headless-наблюдения (в процессе pytest, со стабом `colony_cpp` в песочнице
      и с настоящим расширением в CI);
      (б) `tests/test_trainer_bootstrap.py::test_project_sources_have_no_local_read_before_assignment`
      — AST-проверка по всем модулям проекта; на старом коде падает с указанием
      `watch_champion.py:1181`.
- [x] DoD: `python -m pytest -q` — 24F/334P/67S/2E (падали и падают только тесты без
      собранного `colony_cpp`); CHANGELOG.md обновлён. `./scripts/cpp_checks.sh` в песочнице
      пройден ранее (0 failures), после правок Python C++ не менялся.
- Примечание: попутно исправлен стаб `colony_cpp` в тестах — `minimap()` теперь отдаёт
  `(8, 32, 32)`, как связывание `src/bindings.cpp` (плоский список ломал headless-ветку
  до того, как она доходила до проверяемой строки).

### Этап 1 — контракты и совместимость (1-2 дня) — ✅ СДЕЛАНО 2026-09-24 (кроме ручной проверки UI)
- [x] **P1-1**: «Стоп» в UI — мягкая остановка (`train_ui2/soft_stop.py`, таймаут 60 с,
      fallback `terminate()`, повторное нажатие — сразу); команда → `stop_event`,
      опрашиваемый на каждом шаге среды; при остановке пользователем пропускаются
      eval и турнир (`meta.json: tournament="skipped_user_stop"`); `run_end` пишется
      при любом выходе (`completed/stopped/early_stopped/error/killed`, SIGTERM →
      SystemExit в воркере). Попутно: стоп во время паузы зависал (цикл паузы не
      звал `stop_check`).
- [x] **P1-2**: `dispatch_command`/`process_command_lines` вместо `except (…, Exception)`;
      битая строка, незнакомая команда, `queue.Full` — предупреждение в лог UI;
      недописанная строка файла команд ждёт `\n`.
- [x] **P1-3**: `load_policy_state()` в `rl/_nn_common.py`; переведены `train.py`,
      `train_ui2/worker.py`, `rl/ppo.py`, `train_ui2/evaluator.py`; допускается
      отсутствие только `*_mask_proj`, иначе `ValueError` с путём.
- [x] **P1-4**: `protocol.CMD_*`/`KNOWN_COMMANDS`/`normalize_command`; UI, воркер и
      трейнер берут строки оттуда; AST-тест паритета «UI → trainer»; `else`-ветка.
- [x] **P1-5 (новое, найдено при работе над P1-1)**: `train()` передавал в каждый
      `_collect_rollout` obs первого `reset()` → первый шаг каждого роллаута
      выбирал действие по наблюдению со старта обучения. Фикс:
      `obs = rollout["final_obs"]`; тест непрерывности obs.
- [x] **P2-1**: разбор строки вынесен в `train_ui2/curriculum_table.py::parse_schedule_rows`
      (без Qt); отброшенные строки/повторы — с номером строки в статус-бар и лог;
      голых `except` в `main_window.py` нет (тест); `_curriculum_add_schedule` тоже
      через парсер (падал на битой строке).
- [x] **P2-2**: `get_attr`/`env_is_wrapped` по `indices` (`_get_indices` SB3 +
      проверка границ); попутно `env_method` вызывал метод батча N раз — теперь
      один; частичные `set_attr`/`env_method` — `NotImplementedError`.
- [x] **P2-3**: подмена убрана: ключа нет, `terminal_observation_missing: true` +
      одно предупреждение; бутстрап пропускается. CI-тест
      `test_cpp_vecenv_preserves_terminal_observation` обновлён.
- DoD: новые юнит-тесты зелёные (`tests/test_stop_and_commands.py`,
  `tests/test_policy_load.py`; всего 24F/371P/67S/2E; P2 — `tests/test_review_p2.py` — F только без `colony_cpp`).
  **Осталось вручную на машине с Qt:** кнопки «Стоп/Пауза/Энтропия» (включая
  «Стоп» на паузе и повторный «Стоп»), `--resume-model` на реальном чекпойнте от
  09-21; после «Стоп» в каталоге прогона есть `final_model.pt` + `meta.json` +
  `run_end` в JSONL (сквозной тест `run_train` это покрывает на фейковой среде).

### Этап 2 — наблюдаемость и гигиена ✅ (2026-09-25)
- [x] **P3-1**: `rl/config.py` — импорт `CurriculumState` на верхнем уровне (цикла нет:
      `rl.curriculum` — только stdlib); `TYPE_CHECKING` не годился — ломал
      рантаймовый `typing.get_type_hints`.
- [x] **P3-2**: `{r}` → `{reward}` в `tests/test_reward_clip.py`.
- [x] **P3-5**: `_json_safe` — NaN/±Inf → `null` (включая numpy-скаляры и вложенные).
- [x] **P3-7**: подписи UI/вердикты — «доступно», «окно доступности узкое»,
      «доступно, но не выбирает»; поля протокола `action_legality` не переименованы
      (совместимость); `docs/MONITOR_ACTIONS_2026_09.md` обновлён.
- [x] **P3-4**: разбор `actions.txt` вынесен в `include/colony/watch_ipc.h`
      (`NONE`/`OK`/`INVALID`, без raylib). Невалидное число (вне
      `0..n_actions-1`) потребляется, пишется в `ai_debug_gui.log`, окно отвечает
      `state.json` с `"error"` без шага; недописанный файл по-прежнему не трогается.
      `watch_champion.py` логирует `error`. Проба `tests/cpp/watch_ipc_check.cpp`
      в `cpp_checks.sh`; `gui.cpp` проверен `g++ -fsyntax-only` с заголовками raylib.
- [x] **P3-6**: `ColonyVecEnvCpp::set_terminal_minimap_enabled` (биндинг + фича
      `terminal_minimap_toggle`, не обязательная). `CppVecEnv` выключает её в flat
      (буфер освобождается, `minimap()` на done не считается), `CppVecEnvMinimap` —
      включает. Несовпадение размера → `info["terminal_minimap_missing"]`, и Python
      не кладёт нулевую карту (бутстрап честно пропускается). Попутно: буфер больше
      не обнуляется целиком (n_envs×32 КБ) на каждом шаге, а
      `terminal_minimap_batch` в биндинге не отдаёт неинициализированную память.
      C++: `episode_metrics_check` (вкл/выкл/очистка), Python: `tests/test_review_p3.py`.
- [x] **P3-3**: job `lint` в CI — `ruff check --select F821,F811,F841,F401,E722,B905,E9,F63,F7`
      (ruff 0.16.9), дерево чистое: дубль `spec_for` удалён, 40 неиспользуемых
      импортов, мёртвые локалы, `except:` в `unpack_exe.py`, `zip(strict=…)`.
      Первым же запуском F821 поймал бы `w`, который пришлось вернуть в
      `ui/main_window.py` (кредитный диалог). **mypy в CI не включён**: 175 ошибок в
      18 файлах (`rl`, `train_ui2`) — отдельная задача, как и полный свод ruff + `--fix`.
- [x] **P3-8**: удалены `_get_steps_in_curriculum_stage`, `rollout_time`, локальный
      `loop_detected`; `--command-queue-size` уже подключён в Этапе 1.
- [x] **Loop detector**: метрика на самом деле подключена до UI, но включённый
      детектор **падал** на первом роллауте (`deque[-n:]` → `TypeError`). Исправлено
      (`AsyncTrainer._update_loop_detector`), пункт в `STATE.md`.
- DoD: 24F/383P/67S/2E (те же 24 падения, что и до Этапа 2, — нет `colony_cpp`);
  `./scripts/cpp_checks.sh` зелёный. `bindings.cpp` локально не собирался (нет
  `Python.h`) — проверит job `python-tests`.

### Этап 3 — верификация после сборки `colony_cpp` (в CI или на машине с GPU)
- [ ] `./scripts/cpp_checks.sh` (полный, с `REWARD_STEPS=4000`) — базовая линия.
- [ ] `python -m pytest -q` полностью (в песочнице 24 падения — исключительно из-за
      несобранного расширения; после сборки ожидается 0).
- [ ] Повторить водяной A/B (`scripts/water_ab_run.py` + `water_ab_analyze.py`) уже
      **после** фикса P0-2: сравнение flat/hybrid до фикса содержит неверные value-таргеты
      на усечённых эпизодах.
- [ ] Прогнать `tests/test_watch_visual.py` и новый headless-smoke на машине с Qt/libGL.

### Порядок и причины именно такой
1. P0-1/P0-2 — сегодня, потому что они ломают основной способ смотреть модель и тихо
   искажают обучение в тех самых режимах (minimap/hybrid), которыми сейчас занимаются.
2. P1 — контракты: после них исчезает класс «кнопка ничего не делает молча» и
   «резюм падает на вчерашней модели».
3. P2/P3 — гигиена и наблюдаемость: их можно вести параллельно, они не блокируют обучение.

## Приложение: как воспроизвести ключевые находки

```bash
# P0-1 (статически + рантайм-семантика)
python -m ruff check watch_champion.py --select F821
python watch_champion.py --model-dir <run> --episodes 1 --max-steps 5   # UnboundLocalError (нужен colony_cpp)

# P0-2 (torch, CPU, без среды)
python - <<'PY'
import torch
from rl.rollout_buffer import RolloutBuffer, _TensorRolloutBuffer
mk = lambda cls, **kw: cls(n_steps=4, n_envs=1, n_actions=4, gamma=0.99, gae_lambda=0.95,
                           device=torch.device("cpu"), **kw)
flat = mk(RolloutBuffer, obs_size=3)
tens = mk(_TensorRolloutBuffer, obs_shape=(8, 32, 32))
for buf in (flat, tens):
    for t in range(4):
        obs = torch.zeros(1, 3) if buf is flat else torch.zeros(1, 8, 32, 32)
        buf.add(obs=obs, action=torch.zeros(1, dtype=torch.long), reward=torch.ones(1),
                log_prob=torch.zeros(1), value=torch.zeros(1),
                done=torch.tensor([float(t == 1)]), terminated=torch.tensor([0.0]),
                action_masks=torch.ones(1, 4), trunc_value=torch.tensor([7.0 if t == 1 else 0.0]))
print("flat  :", flat.trunc_values.tolist())   # [0.0, 7.0, 0.0, 0.0]
print("tensor:", tens.trunc_values.tolist())   # [0.0, 0.0, 0.0, 0.0]  <-- баг
PY

# Прочее
python -m ruff check . --select F821,F811,E722,F841          # undefined names, дубли, бар-эксепты
python -m pytest -q --continue-on-collection-errors -rf      # в песочнице: 24F/323P/67S/2E (все F — нет colony_cpp)
./scripts/cpp_checks.sh                                      # C++-пробы: ALL CHECKS PASSED
```

Ограничения ревью: `colony_cpp` не собран (нет `Python.h`), Qt-модули не импортируются
(нет `libGL.so.1`), поэтому GUI/IPC-пути и всё, что зависит от реальной среды, проверены
только статически; обучение/награды в песочнице не измерялись.
