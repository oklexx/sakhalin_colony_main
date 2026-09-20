# Changelog

Формат: `## [дата]` + список изменений. Завершённые задачи из [STATE.md](STATE.md)
переносятся сюда. Источники: `docs/*_2026_09.md`, аудиты, README §11.

## [2026-09-20] — два пользовательских бага: падение обучения в `hybrid` и «молчаливое» GUI-наблюдение

Оба бага пришли из эксплуатации (Windows, `C:\Colony`): обучение падало на
первом шаге, а «👁 Наблюдать» с галочкой **GUI-окно** не давал ни окна, ни
строки в логе. Ниже — причины, правки и чем они закреплены.

### Баг 1 — `TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple`

- **Причина.** Трассировка: `worker.py` → `rl/async_trainer.py::train` →
  `_collect_rollout` → `rl/actor_critic_hybrid.py::get_value` → `forward` →
  `self.flat_trunk(flat_in)`. Режим наблюдения читался как
  `getattr(self.em, "obs_mode", "flat")`, а у `EnvManager` такого атрибута не
  было: дефолт `"flat"` молча включался при реальном `hybrid`, и в `Linear`
  уезжала **пара** `(flat, minimap)` вместо тензора. Падение — на первом же
  шаге обучения, до всякого прогресса.
- **Правки.** `rl/env_manager.py`: `self.obs_mode` — публичный атрибут,
  выставленный в `__init__` из `cfg.obs_mode` (раньше атрибута не было вовсе,
  поэтому `getattr` и срабатывал). `rl/async_trainer.py`:
  `_obs_mode()` (единственный способ спросить режим) и `_split_hybrid_obs()`
  (разбор пары с явной ошибкой, если форма не та); маски передаются ключевыми
  аргументами, чтобы перепутать порядок было нельзя. `rl/actor_critic_hybrid.py`:
  `_split_obs()` — сеть сама разбирает и тензор, и пару, поэтому
  `forward/get_value/get_actions` больше не зависят от того, кто и как передал
  наблюдение.
- **Регрессия.** `tests/test_trainer_bootstrap.py` (9 тестов): реальный
  `EnvManager` + реальная `ActorCriticHybrid` + реальный
  `AsyncTrainer._collect_rollout` на фейковом vecenv — прогон до первого шага в
  `flat`/`minimap`/`hybrid`, форма масок, разбор пары, а также
  `test_async_trainer_global_names_resolve` (см. ниже). Проверено в обе
  стороны: со старым кодом 6 тестов падают, с новым — все зелёные.

### Баг 2 — GUI-окно не запускалось, лог пустой

Диагноз ставился удалённо и точно установлен не был; вместо guesses закрыт весь
класс «ничего не происходит». Протокол и тексты сообщений — в новом
[docs/GUI_WATCH_2026_09.md](docs/GUI_WATCH_2026_09.md).

- **Бесконечные ожидания → таймауты + heartbeat.** `wait_for_state()`
  (свежесть `state.json` по mtime_ns+размеру, lock-step: действие считается
  только из состояния, порождённого ПРЕДЫДУЩим действием), `GUI_START_TIMEOUT`
  25 с, `GUI_STEP_TIMEOUT` 30 с, `GUI_RESET_TIMEOUT` 20 с,
  `GUI_HEARTBEAT_EVERY` 5 с — в лог UI пишется «жду первое состояние от
  GUI-окна… 5 с» вместо тишины.
- **Бесконечный рестарт exe → `MAX_GUI_RESTARTS = 3`** и диагноз
  `describe_gui_failure()`: код возврата (с подсказкой `0xC0000135` = не найден
  DLL), путь exe, IPC-каталог и **хвост `gui_output.log`** — stdout/stderr окна
  теперь перехватываются (консоль у watch-процесса скрыта `CREATE_NO_WINDOW`).
- **Окна-сироты → per-run IPC + убийство дерева.** Каталог
  `%TEMP%/colony_watch/<pid>_<мс>` свой на каждый запуск и удаляется по
  завершении (`make_ipc_dir()`); `_stop_watch_proc()` в UI бьёт по дереву
  (`taskkill /T /F`), а `_check_watch_exit()` сообщает о завершении воркера.
  Раньше пережившее драйвер окно доедало чужой `actions.txt`, и следующий
  запуск общался с сиротой — своё окно выглядело мёртвым. Метка каталога
  `<pid>_<мс>` дополнена счётчиком вызовов (`_s<N>`): два запуска в одном
  процессе, попавшие в одну миллисекунду, получали ОДИН путь, если предыдущий
  каталог уже был убран (поймано собственным тестом — плавало примерно в одном
  прогоне из пяти). На POSIX тот же
  эффект даёт `start_new_session=True` + `os.killpg(SIGTERM)` (одиночный
  `kill()` ребёнка не доставал), а ручка лог-файла наблюдения теперь
  закрывается в родителе сразу после `Popen` — раньше каждый запуск наблюдения
  оставлял в UI открытый fd.
- **Гонки IPC → атомарность с обеих сторон.** `src/gui.cpp`: `state.json`
  (~100 КБ с миникартой) пишется в `.tmp` и переименовывается (5 попыток,
  затем `copy_file(overwrite_existing)` — на Windows rename не проходит, пока
  читатель держит файл); `actions.txt` **не удаляется**, если разобрать его не
  удалось (удаление означало бы «прочитано» → взаимное ожидание навсегда);
  `remove()` через `error_code`, чтобы гонка не роняла окно.
  `watch_champion.py::write_action` — тоже tmp+rename.
- **Протухший exe.** `gui_exe_stale_sources()` сравнивает mtime exe с
  `src/*.cpp`/`include/**` и пишет `WARNING: GUI exe собран ДО правок в …` —
  старый exe не понимает `--tax-to-debt`/`--curriculum`/миникарту и молча
  играет в другую игру. `find_gui_exe()` ищет по 7 каталогам × 3 имени и
  уважает `COLONY_GUI_EXE`.
- **Выбор файла весов: UI и CLI согласованы, порядок чекпойнтов числовой.**
  `watch_champion.resolve_model_file()` вынесен из `main()` (запрошенный →
  `final_model.pt` → `best_model.pt` → свежайший чекпойнт): старая цепочка
  знала только final и чекпойнты, поэтому `--model-file final_model.pt` на
  прерванном прогоне грузил произвольный чекпойнт вместо чемпиона турнира.
  «Свежайший» теперь определяется по **числу шагов**
  (`train_ui2.models.checkpoint_sort_key`): `sorted()` по имени возвращал
  `checkpoint_999999_steps.pt` вместо `checkpoint_1000000_steps.pt` — то же
  исправлено для sidecar-нормализаций `checkpoint_*_steps.norm.json`
  (`loose=False`, чтобы вместо нормализации не подсовывались веса).
- **Кнопка «ничего не делала»: модель без `final_model.pt` была невидима.**
  `final_model.pt` пишется только в конце обучения, поэтому у остановленного
  или упавшего прогона есть лишь `best_model.pt`/чекпойнты — а
  `train_ui2/models.py::scan()` требовал именно `final_model.pt`, и «👁
  Наблюдать» отвечал «Нет моделей — сначала обучите». Теперь
  `MODEL_WEIGHT_NAMES = ("final_model.pt", "best_model.pt")` +
  `checkpoint_*_steps.pt` (`has_model_weights()`, `pick_model_file()`),
  `_finetune_selected` берёт файл через `pick_model_file()`, а пустой список
  сообщает в диалоге и в логе, **где** искали (`registry.root`) и что считается
  весами.
- **Дубли в панели лога.** Каждое сообщение наблюдения печаталось И
  отправлялось JSON-ом (`print` + `emit_log`) — строка «Game Over» выглядела как
  две. Теперь `say()`/`_log()` выбирают один канал; заодно устранено последнее
  такое место (`ERROR: GUI exe not found`).
- **Регрессия.** `tests/test_watch_visual.py` (18 тестов): настоящий
  `watch_champion.py` против **фейкового окна** (Python-скрипт в роли exe) в
  режимах `ok / legacy / die / silent / freeze / no-minimap` — счастливый путь,
  legacy-бутстрап действием 0 (DAY), мёртвый exe, молчащее окно (handshake +
  heartbeat), замершее окно (step-timeout), `hybrid` без миникарты, per-run IPC
  и его уборка, независимость двух запусков (включая попадание в одну
  миллисекунду), атомарность `write_action`, freshness lock-step, детект
  протухшего exe, параметры `Popen` (`--headless-ai`, закрытая ручка лога),
  авто-рестарт карты после 2 эпизодов, цепочка `resolve_model_file`.
  Стаб `colony_cpp` ставится только на время модуля и убирается за собой.
  Плюс `tests/test_models.py`: +6 тестов на `scan()`, `pick_model_file()` и
  числовой порядок чекпойнтов.

### Попутно — молчаливые `NameError` в `rl/async_trainer.py`

Нашлись статически (ruff F821), пока искался баг 1; оба глотались
`except Exception: pass`, поэтому не проявлялись ни в логе, ни в тестах:

- `json.dump(...)` при записи `checkpoint_*.meta.json` — `json` был импортирован
  локально в ДРУГИХ методах. Итог: чекпойнты оставались **без meta**, а
  watch/турнир не могли восстановить стадию курикулума, под которой чекпойнт
  учился. `import json` поднят на уровень модуля.
- `norm_str` в турнире чекпойнтов (end-of-training tournament) — имя не
  определено в этой области; кандидат без `.norm.json` молча пропускался.
  Заменено на существующий `norm_path` финальной модели.
- Третий экземпляр того же класса нашёлся в `watch_champion.py`:
  `except (json.JSONDecodeError, OSError)` в `main()` при локальном
  `import json as _json` — битый `best_model.meta.json` ронял наблюдение с
  NameError ровно там, где должен был аккуратно пропустить файл. `json` и
  `subprocess` подняты на уровень модуля (последний был нужен ещё и аннотации
  `-> "subprocess.Popen"` в `launch_visual_watch`).
- Гард: `test_module_global_names_resolve` (parametrize: `rl/async_trainer.py`,
  `watch_champion.py`) проходит по `symtable` исходника — без импорта модуля,
  поэтому torch/`colony_cpp` не нужны — и требует, чтобы каждое глобальное имя
  было связано в модуле или в builtins. Проверено в обе стороны: без
  `import json` тест перечисляет все три области `async_trainer`, где `json` не
  определён, и `watch_champion.main: json`.

### Документация

- Новый **`docs/GUI_WATCH_2026_09.md`**: участники наблюдения, IPC-контракт
  (`actions.txt`/`state.json`/`gui_output.log`, атомарность, кто что удаляет),
  флаги окна, таблица таймаутов, таблица «симптом → сообщение в логе → причина»,
  что считается моделью, ручная отладка (`COLONY_GUI_EXE`, `%TEMP%\colony_watch`),
  ограничения.
- **README.md**: §0 — счётчики (43 тест-файла + 22 C++-пробы) и ссылка на новый
  док; §1 — запуск `--visual`; §2 — карта проекта (роль `watch_champion.py`,
  `train_ui2/models.py`, 6 вкладок UI); §3 — таблица контракта `obs_mode`
  (причина бага 1); §5.1 — **49** действий вместо 45 (блок 45..48 ROAD_E/W/S/N
  отсутствовал) и маски направлений; §5.3 — миникарта **глобальная 8×32×32**
  (было «[8, 29, 29], окно радиуса R=14»); §8.2 — что считается моделью и зачем
  `checkpoint_*.meta.json`; §9.1 — вкладка «Наблюдение» (галочка, kill-дерева,
  диагностика); §10 — команды для новых регрессий и честно про поведение без
  собранного `colony_cpp`.
- **RULES.md**: золотая синхронизация — п. 5 уточнён (49 действий, инвариант
  `A_BUILD0 + i ↔ build_ids_[i]`), добавлены п. 6 (контракт `obs_mode`, запрет
  `getattr(em, "obs_mode", "flat")`) и п. 7 (IPC наблюдения: атомарность,
  per-run каталог, таймауты вместо `while True: sleep()`); в Python-правилах —
  почему `NameError` внутри `except Exception: pass` опасен и каким тестом
  закрыт; в тестах — стабы в `sys.modules` ставить только на время своего модуля;
  в стиле ошибок — «ничего не происходит» тоже ошибка; опечатка «дегра
  gracefully» исправлена.
- **STATE.md**: статус на 2026-09-20, закрытые задачи дня, новые пункты
  (длинный прогон `hybrid`/`minimap`, обратная связь по GUI-наблюдению),
  фактический размер `src/env.cpp` (1140 строк, остаток — `step()` ~810) и
  ограничения наблюдения.

### Проверки

- `tests/test_watch_visual.py` 18/18, `tests/test_trainer_bootstrap.py` 10/10,
  `tests/test_models.py` 22/22 (5 прогонов подряд без единого падения, в том
  числе под нагрузкой 6 процессов на 2 CPU).
- Полный `pytest` в песочнице (без собранного `colony_cpp` и без CUDA):
  список падений побайтово совпадает с эталоном ДО правок (24 падения —
  `ModuleNotFoundError`/`NameError: colony_cpp`, то есть окружение), новых
  падений нет; passed вырос с 228 до 262 за счёт 34 новых тестов.
- `./scripts/cpp_checks.sh` — все проверки прошли (22 пробы скомпилированы,
  7 проверок зелёные); `g++ -fsyntax-only src/gui.cpp` — чисто.
- **CI (GitHub Actions) — первый зелёный прогон в истории проекта** (PR #13):
  `C++ пробы (compile-all + проверки)` — pass 1m28s, `colony_cpp + pytest` —
  pass 3m15s (сборка `colony_cpp` через CMake, handshake, полный pytest с
  Qt-библиотеками). До этого `.github/workflows/ci.yml` не запускался ни разу,
  а сборка расширения и Qt-тесты в песочнице непроверяемы.

## [2026-09-19] — план docs/REMAINING_WORK_2026_09.md: P0, P1, P2

План работ создан в репозитории (`docs/REMAINING_WORK_2026_09.md`) — раньше он
жил только в переписке. Статус каждого пункта — с измерениями.

### P0 — влияет на обучение
- **P0-1 Диагностика маски WaterChannel: легальность корректна.** Новый зонд
  `tests/cpp/water_mask_check.cpp`: бит маски BUILD_WaterChannel совпал с
  полным перебором 200×200 (`can_build_at`) в **457/457** состояний,
  `find_lot` не теряет и не выдумывает ячейки. Маска закрыта из-за денег
  (240 состояний) и отсутствия легальной ячейки (210), курикулум — 0.
  Найден остаточный дефект **наведения**, а не легальности: на seed 100
  `find_lot_dir` «перескакивает» цель (дистанция 7.81 → 19.4), 4 направленных
  действия не покрывают диагональ.
- **P0-2 Верификация `reward_v4` на длинном прогоне.** Новый зонд
  `tests/cpp/reward_v4_longrun.cpp` (5 политик × 3 сида × 3 профиля ×
  4000 шагов): на v4 `greedy_mix` = **+1830.0** против `road_spam` = −297.4 →
  **v4 ломает road-spam-аргмакс**; 8 водоканалов построено. Исторический +898
  не воспроизводится ни на одном профиле — дыру закрыл R1 (дороги исключены из
  `milestone_base_bonus`), а не v4.
- **Исправлен гибрид в `configs/reward_v3.json`**: профиль не задавал
  v4-терминалы (`buy_food_penalty`, `goal_survival_coeff`,
  `main_tax_cash_bonus`, `main_tax_pressure_coeff`) и молча наследовал v4 из
  дефолтов структуры — A/B «v3 против v4» сравнивал гибрид. Теперь
  `v3_file == v3_pure` (проверяется зондом).
- **P0-3 Уменьшенные exp-конфиги** под 12/24 ГБ: `exp_25_minimap_n64`
  (4.29 ГБ), `exp_26_hybrid_n64` (4.45), `exp_26_hybrid_n128` (8.90),
  `exp_27_minimap_lr_low_n64` (4.29). Пересчёт показал, что README занижал
  оценку: оригиналы с `n_envs=1024` — 68.7/71.3 ГБ, а не «56+». Гард —
  `tests/test_exp_configs.py`.

### P1 — гигиена
- **P1-4 README**: убраны живые ссылки на удалённые файлы (`auto_trainer.py`),
  `REPORT.md/REPORT_2026_09.md` → `docs/RL_DIAGNOSIS_2026_09.md`, «42 поля
  наград / reward_v3.json» → 55 полей / `reward_v4.json`, миникарта
  `8×29×29 = 6728` → `8×32×32 = 8192` (2 места), «tests/ 28 файлов» → 41 + 21
  C++-проба, добавлен раздел про CI. Дрейф подписей в `train_ui2/main_window.py`
  («профиль v3» при каноне v4) устранён: `_reset_rewards_canonical`
  (алиасы `_v3`/`_v2` сохранены для теста).
- **P1-5 `tests/test_ui2_smoke.py` починен, а не удалён**: файл был скриптом
  (проверки при импорте + `sys.exit(1)`) → под pytest падал на сборке и писал в
  настоящий `~/colony_runs/`. Переведён в pytest-форму (маркер `ui`, `HOME` →
  `tmp_path`, skip без Qt); `--ignore` из `pytest.ini` убран, причина
  задокументирована.
- **P1-6 CI**: `.github/workflows/ci.yml` (job `cpp-probes` — только g++;
  job `python-tests` — CMake-сборка `colony_cpp`, handshake, полный pytest с
  Qt-библиотеками и `QT_QPA_PLATFORM=offscreen`) + локальный эквивалент
  `scripts/cpp_checks.sh` (компиляция всех 21 пробы и 7 проверок с кодом
  возврата). Добавлен `tests/conftest.py`: `sys.path` больше не зависит от
  порядка сбора тестов.

### P2 — техдолг плана ревью
- **P2-8** `set_minimap_radius` (обе привязки) выдаёт `DeprecationWarning`:
  миникарта глобальная 32×32, радиус на наблюдение не влияет.
- **P2-9** Магические числа road-shaping вынесены в `RewardConfig`
  (6 полей: `road_shaping_cap`, `road_shaping_per_cell`, `water_reach_bonus`,
  `water_reach_radius`, `road_no_progress_penalty`, `road_progress_epsilon`);
  значения = прежний хардкод, `reward_v4_longrun` даёт те же числа. Поля
  добавлены в `rl/config.py`, `configs/reward_v4.json`, биндинги и UI
  (группа «Дороги к воде»).
- **P2-10** `Environment::lot_ok` делегирует `Game::can_build_at` — устранены
  два расхождения legacy-версии (`destroyed_lots` и Roads в правиле
  `no_near_base`).
- **P2-11** Кеш легальности лота внутри одного вызова `action_mask`
  (ключ `(need_earth, no_near_base)`): −61 % на reset, −36 % на 50 дорогах,
  −27 % на 200 дорогах с деньгами, −21 % в среднем (бенч
  `tests/cpp/action_mask_bench.cpp`, 2000 вызовов, `-O2`); число открытых
  BUILD-действий идентично до/после (11/14/1/25).

### Тесты
- `tests/test_reward_field_sync.py` — новая стража золотого правила на уровне
  исходников: C++ `RewardConfig` ↔ `rl.config.RewardConfig` ↔ биндинги ↔
  `configs/reward_v4.json` ↔ `configs/reward_v3.json` ↔ UI (`REWARD_SPECS` +
  `REWARD_FLAGS`). Она же нашла реальный дрейф: два C++-флага-абляции
  (`priority_count_over_allowed`, `obs_mask_locked_catalog`) отсутствовали в
  Python-dataclass, поэтому `RewardConfig.from_dict` молча выбрасывал их из
  reward-JSON — флаги добавлены в dataclass, профиль и UI.
- `tests/test_reward_default_profile.py::test_ui_reward_specs_cover_profile`
  больше не скипается молча: импортировал удалённый `train_ui`, переведён на
  `train_ui2`.
- `tests/test_colony_robustness.py::test_reward_config_completeness` сверяет
  dataclass с каноническим профилем по ключам и значениям вместо хардкода
  «47 полей».
- Прогон: `./scripts/cpp_checks.sh` — **все проверки прошли** (21 проба
  скомпилирована, 7 проверок, 0 fail); `pytest` — **228 passed, 66 skipped**,
  список падений идентичен baseline `a2ee3d0` (22 failed + 2 ошибки сбора —
  отсутствие собранного `colony_cpp` и CUDA в песочнице; в CI они есть).

### Не сделано
- **P2-7 распил `src/env.cpp`** — отдельным PR (план и порядок выноса в
  `docs/REMAINING_WORK_2026_09.md`).
- Прогон обучения на GPU и первый прогон CI — требуют окружения, которого в
  песочнице нет (нет GPU, нет заголовков CPython и библиотек Qt).

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
