# Sakhalin Colony — RL-обучение экономической стратегии (PPO)

Обучение агента PPO игре в экономическую стратегию «колония на Сахалине» (1890 год): агент строит здания, manages ресурсы, платит налоги и выживает до 10 000 игровых дней. Среда — C++ (`colony_cpp.pyd` через pybind11), обучение — PyTorch на GPU, есть GUI-дашборд на PySide6.

> **Отчёты по проекту:** [REPORT.md](REPORT.md), [REPORT_2026_09.md](REPORT_2026_09.md), [TRAINING_REPORT.md](TRAINING_REPORT.md), [REVIEW_REPORT.md](REVIEW_REPORT.md).

---

## 1. Быстрый старт

```bash
# 1. Зависимости (Python 3.10+, CUDA GPU)
pip install -r requirements.txt        # torch, numpy, tensorboard, gymnasium, pybind11, stable-baselines3, PySide6, pyqtgraph

# 2. Сборка C++-среды (нужны CMake 3.20+ и компилятор C++17; на Windows — MSVC)
build_pyext.bat                        # Windows: -> python/colony_cpp.pyd
# или вручную: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release

# 3. Проверка (расширение обязано отвечать на handshake)
python -c "import sys; sys.path.insert(0,'python'); import colony_cpp; print(colony_cpp.extension_info())"

# 4. Обучение (flat-режим, стабильные параметры)
python train.py --steps 2000000 --envs 8 --n-epochs 4 --ent-coef 0.015 \
    --obs-mode flat --eval-freq 500000 --eval-episodes 10 --name my_run

# 5. GUI-дашборд обучения
python run_train_ui2.py

# 6. Посмотреть, как играет чемпион
python watch_champion.py --model-dir ~/colony_runs/models/my_run
```

Артефакты обучения: `~/colony_runs/models/<name>/` (чекпойнты), `~/colony_runs/logs/<name>/` (TensorBoard: `tensorboard --logdir ~/colony_runs/logs`).

---

## 2. Карта проекта — «что где лежит»

```
sakhalin_colony_main/
├── train.py                  # CLI-точка входа обучения (аргументы -> Config -> EnvManager -> AsyncTrainer)
├── auto_trainer.py           # Optuna-поиск гиперпараметров (objective = best_score из best_model.meta.json)
├── training_cycle.py         # Пакетный цикл: 30 прогонов × 10M шагов с отчётом
├── observe.py                # Прогон чекпойнта с подробным логом шагов/наблюдений
├── watch_champion.py         # Визуальный просмотр игры чемпиона в реальном времени
├── sweep2..sweep7_*.py, reward_sweep.py  # Пакетные свипы наград через train_ui2/worker.py
├── CMakeLists.txt            # Сборка colony_cpp (.pyd кладётся в python/)
├── requirements.txt
│
├── rl/                       # ★ Python-слой обучения
│   ├── config.py             #   Config + RewardConfig (42 поля наград; ДЕФОЛТЫ = C++ = reward_v3.json)
│   ├── actor_critic.py       #   MLP 256×256 для flat-obs (131K параметров)
│   ├── actor_critic_cnn.py   #   CNN для minimap-obs
│   ├── actor_critic_hybrid.py#   CNN+MLP для hybrid-obs
│   ├── rollout_buffer.py     #   GPU-буфер rollout + GAE (RolloutBuffer / _TensorRolloutBuffer)
│   ├── ppo.py                #   PPO: collect_step (сбор), update (обучение), AMP, torch.compile
│   ├── env_manager.py        #   Связка CppVecEnv + модель + буфер + PPO; curriculum
│   ├── async_trainer.py      #   Главный цикл: rollout -> update -> eval -> checkpoint (SyncTrainer по факту)
│   └── loop_detector.py      #   Детектор зацикливания действий (по умолчанию ВЫКЛЮЧЕН)
│
├── python/                   # ★ Python-обёртки C++
│   ├── cpp_vecenv.py         #   CppVecEnv: батч из N сред, маски, terminated, нормализация, сиды
│   ├── cpp_env.py            #   CppColonyEnv: одиночная gym-среда (для eval/UI) + Normalizer
│   ├── minimap.py            #   MinimapVecEnvWrapper / MinimapSingleEnvWrapper (8×29×29)
│   └── colony_cpp.pyd        #   Собранный C++-модуль (результат cmake; НЕ в git — см. build_pyext.bat)
│
├── src/                      # ★ C++-ядро (namespace colony)
│   ├── env.cpp / env.h       #   ColonyEnvCpp (одна среда): obs, step, награды, маски, curriculum
│   │                         #   ColonyVecEnvCpp (батч): потоки, obs_rms_, rew_rms_, авто-ресет
│   ├── game.cpp / game.h     #   Игровая логика: дни, сезоны, рождение/смерть, стройка, рынок, налоги
│   ├── earth.cpp / earth.h   #   Карта: лоты (земля/вода/лес/уголь/железо/нефть/золото), BFS-поиск участка
│   ├── data.cpp / bases.h    #   BaseData из bases.json (33 здания), BaseEvent из events.json
│   ├── rng.cpp / rng.h       #   MtRandom / PCG64 — детерминированный RNG по сиду
│   ├── running_mean_std.h    #   RunningMeanStd (нормализация obs/reward, JSON save/load)
│   ├── thread_pool.h         #   Пул потоков для параллельного step N сред
│   ├── bindings.cpp          #   pybind11-биндинги (colony_cpp)
│   ├── gui.cpp, main.cpp     #   Нативный GUI/демо (не нужны для RL)
│   └── constants.h           #   Все константы: MAX_STEPS=10000, TAX_GRACE_DAYS=60, ...
│
├── configs/
│   ├── bases.json            #   33 здания: цена, время стройки, рабочие, сезоны, потребление/прибыль
│   ├── events.json           #   Случайные события (аварии на шахтах/нефтянках)
│   ├── reward_v3.json        #   ★ Канонический профиль наград (= дефолты кода)
│   ├── reward.json, reward_v2.json  # Исторические профили (только явный --reward-config)
│   └── exp_25/26/27_*.json   #   Экспериментальные наборы гиперпараметров (minimap/hybrid)
│
├── train_ui2/                # PySide6-дашборд 2.0 (запуск: python run_train_ui2.py)
│   ├── app.py, main_window.py  #   Вкладки Обучение/Мониторинг/Награды/Модели/Наблюдение
│   ├── worker.py             #   Обучение в отдельном процессе (JSONL-протокол)
│   ├── protocol.py           #   Сообщения ready/log/progress/saved/done/error/command
│   ├── evaluator.py          #   ★ run_eval(): оценка чекпойнта (argmax, с нормализацией)
│   ├── parameter_widget.py   #   ParamSpec всех параметров (дефолты = rl/config.py)
│   ├── controls.py, charts.py, theme.py, models.py
│
└── tests/                    # 28 файлов: test_gae, test_ppo_smoke, test_evaluator, test_reward_clip,
                              # test_milestones, test_curriculum, test_normalizer, bench_per_step, ...
```

---

## 3. Архитектура и поток данных

```
train.py (CLI)
   └─> Config / RewardConfig (rl/config.py)
   └─> EnvManager (rl/env_manager.py)
         ├─> CppVecEnv (python/cpp_vecenv.py) ──> colony_cpp.ColonyVecEnvCpp (src/env.cpp)
         │       N параллельных Game-инстансов, thread_pool, obs/reward-нормализация (RMS)
         ├─> ActorCritic / CNN / Hybrid (rl/actor_critic*.py)  [GPU]
         ├─> RolloutBuffer (rl/rollout_buffer.py)              [GPU]
         └─> PPO (rl/ppo.py)                                   [GPU]
   └─> AsyncTrainer.train() (rl/async_trainer.py)
         цикл: _collect_rollout (n_steps×n_envs шагов) -> ppo.update (n_epochs×батчи)
               -> [eval каждые eval_freq] -> [checkpoint каждые save_freq] -> [curriculum]
```

**Один шаг обучения** (`EnvManager.collect_step`, rl/env_manager.py:255-310):
1. `env.action_masks` → тензор [n_envs, 45] (маски обновляются в C++ после каждого step_wait);
2. `PPO.collect_step`: forward (no_grad, autocast) → маскировка логитов (`-inf`) → Categorical → action, log_prob, value;
3. `action_gpu.cpu()` → `env.step_async/step_wait` (C++ выполняет N шагов параллельно);
4. C++ возвращает obs (нормализованные), rewards (нормализованные), dones, `_last_terminateds`, infos;
5. `buffer.add(obs, action, reward, log_prob, value, done, terminated, masks)` — всё на GPU.

**Update** (`PPO.update`, rl/ppo.py): `compute_gae` (рекурсия с конца, маска бутстрапа = terminated, truncation НЕ терминалится) → нормализация advantages → n_epochs × minibatch: clip-loss + 0.5·MSE(value) − ent·entropy, grad-clip 0.5, Adam.

---

## 4. Игровая механика (C++)

### 4.1 Мир и время
* Карта `map_size×map_size` (дефолт 280; эксперименты 200). Лоты: обычная земля, вода, лес, уголь, железо, нефть, золото (`earth.cpp`).
* Старт: 01.03.1890, **82 000 денег**, **28 жителей** (`constants.h:15-17`).
* Любое действие продвигает время: BUILD/MGR-действия = 1 день, `DAY` = 1 день, `WEEK` = 7 дней.
* Сезоны (весна/лето/осень/зима) влияют на работу зданий (`work_seasons` в bases.json).
* Лимит эпизода: **MAX_STEPS = 10 000 действий** → truncation (не терминал; GAE бутстрапится).

### 4.2 Здания (configs/bases.json — 33 записи, 32 строящихся + City)
`City` (DEPOT_ID) — стартовая база, не строится. Ключевые поля: `price`, `build_time` (дней), `live_years` (износ → нужен REPAIR), `home_places` (жильё), `need_workers`, `need_earth` (площадь), `work_seasons`, `consume`/`profit` (цепочки производства), `style` (`no_near_base`, `no_preserve`, `no_occupy`).

Примеры: Farm 19 814 (вода→еда), Coalmine 25 471 (дерево+энергия→уголь+камень), PowerStation 92 433 (уголь→энергия), House 14 038 (60 мест жилья), SuperHouse 971 000 (4000 мест), AtomStation 120 000 000 (620 энергии).

### 4.3 Ресурсы («сундук», 9 типов, constants.h:10-11)
gold, food, water, coal, iron, oil, stone, wood, energy. Здания потребляют/производят ежедневно; избыток > 200 можно продавать (`SELL_SURPLUS`), еду докупать (`BUY_FOOD`).

### 4.4 Население
Рождение через BIRTH_DAYS=2184 дня жизни, смерть через DEATH_DAYS=25 480; прибытие партий (ADDPEOPLE=28±5). Нужны жильё (home_places) и рабочие места; переполнение жилья → штраф.

### 4.5 Налоги, кредит, game over
* Налоги: годовой и «главный» (NALOG_* в constants.h); неуплата → −5 за шаг; просрочка **TAX_GRACE_DAYS=60** → game over.
* Кредит: TAKE_LOAN (+50 000, лимит 100 000... max_credit), 2% (CREDITPERCENT); превышение лимита → game over.
* Game over также по `game_over()` игры (коллапс колонии). Все три ветки: `terminated=true` + штраф `game_over_penalty` (env.cpp:1095-1101).

### 4.6 События (configs/events.json)
Раз в ~EVENT_ATTEMPT_DAYS=455 дней — авария на случайной шахте/нефтянке (чинится REPAIR).

---

## 5. Пространства действий и наблюдений

### 5.1 Действия — 45 дискретных (constants.h:126-129)
```
0  DAY                      1  WEEK
2..33  BUILD_* (32 здания в порядке bases.json, кроме City):
       Farm, Garden, WaterChannel, Sawmill, Coalmine, Ironmine, Refinery, Goldmine,
       PowerStation, HydroStation, Road (лимит 25), House, SmallHouse, Fish, CoalCut,
       HuntingLand, CowFarm, Mushroom, BigHouse, BigFarm, Apiary, Torchlight, Hothouse,
       SuperHouse, BigSawmill, WaterMill, BigRefinary, Puerperal, BigIronmine,
       AirStation, SmallAtomStation, AtomStation
34..44  MANAGER (env.cpp, имена в python/cpp_env.py:174-177):
       IMPROVE_LAND, REPAIR, REPAIR_ALL, DEMOLISH, PRESERVE, UNPRESERVE,
       SELL_SURPLUS, BUY_FOOD, TAKE_LOAN, REPAY_LOAN, PAY_TAX
```
**Маски:** `DAY`/`WEEK` доступны всегда (env.cpp:562-564); BUILD — если разблокировано curriculum, есть деньги, есть участок (BFS с учётом дорог/соседства), для Road — лимит 25.

### 5.2 Наблюдение flat — 246 чисел (env.cpp, формула в env.h: obs_size())
Категории (все нормализованы масштабом, затем RunningMeanStd, клип ±10):
время (год/месяц/день/сезон) → деньги/кредит/население/занятые → 9 ресурсов → жильё/рабочие/свободные/переполнение → флаги и суммы налогов → days_alive → curriculum_stage → **32 счётчика зданий** → статистика износа/стройки/preserve → 9 цен продажи → категории → **9 балансов ресурсов** (производство−потребление) → 32 флага простоя по типам.

### 5.3 Minimap — тензор [8, 29, 29] (env.cpp:651-690)
Окно радиуса R=14 вокруг старта колонии. Каналы: 0 земля, 1 вода, 2 лес, 3 уголь, 4 железо, 5 нефть, 6 золото, 7 занято постройкой. Режимы: `--obs-mode flat|minimap|hybrid` (hybrid = minimap + flat, модели в `rl/actor_critic_*.py`).

---

## 6. Система наград — ПОЛНАЯ карта

### 6.1 Конфигурируемые веса (RewardConfig: rl/config.py, src/env.h, configs/reward_v3.json — все источники СОГЛАСОВАНЫ; номера строк env.cpp приблизительные)

| Вес | Дефолт | Где применяется (env.cpp) | Формула |
|---|---|---|---|
| build_bonus | 2.0 | step() | `2 + log2(1 + ypv/1000)` за постройку |
| build_cost_penalty | 0.0001 | :804 | `−0.0001 × цена` |
| diversity_bonus | 3.0 | step() | за каждый новый тип здания |
| proximity_bonus | 0.5 | :824 | рядом с ресурсом |
| provider/prereq бонусы | (вкл/выкл флагом) | :828-835 | за обеспечение цепочек |
| chain_bonus | 1.0 | :1037 | `× log2(1 + prod/1000)` потребителю |
| chain_daily | 0.5 | step() | ежедневно за активную цепочку |
| first_extraction_bonus | 3.0 | step() | разово за первую добычу типа ресурса (×вес) |
| extraction_daily | 0.3 | step() | ежедневно за активную добычу (×вес×насыщение) |
| need_fill_bonus | 1.5 | step() | за производителя «голодающего» ресурса |
| loan_penalty | 0.5 | step() | стоимость TAKE_LOAN (вычитается) |
| preserve_penalty | 0.3 | step() | плата preserve/unpreserve (вычитается) |
| buy_food_penalty | 3.0 | step() | штраф за BUY_FOOD (вычитается) |
| daily_income | 1.0 | :1045 | `× log1p(daily_total/100)` |
| novelty | 5.0 | step() | первый ЗАРАБОТАВШИЙ нового типа |
| sale_bonus | 0.5 | step() | `× log1p(sale/100)` |
| tax_daily_bonus | 0.3 | :726 | день без налогов |
| error_penalty | −2.0 | ~15 мест | любое неудачное действие |
| demolish_penalty | −3.0 | :861 | за снос |
| manual_tax_penalty | −0.5 | step() | ручная уплата налога (штраф) |
| survival_bonus | 0.0 | :1106 | пассивное выживание (выключено) |
| survival_coeff | 0.0 | step() | `× Δnet_worth` (выключено) |
| idle_build_penalty | −2.0 | step() | простой без стройки дольше 7 дней |
| milestone_* | 30/2/2/5 | :951 | каждые 5 баз / 50 людей / 100 дней / год |
| game_over_penalty | 10.0 | :1095-1101 | `rew −= 10` при game over |
| clip_reward_min/max | ±50 | :1146 | клип сырой награды |

### 6.2 Бывшие хардкоды — теперь настраиваются (RewardConfig)
Неуплата налога −5/день (tax_fail_penalty) · долг −0.1×credit/1000 (debt_coeff) · рождение/прибытие +1 (born_bonus) · **смерть −20** (death_penalty) · потеря базы −30 (base_lost_penalty) · переполнение жилья −2 (home_overflow_penalty) · жильё при нехватке 3×log1p (housing_need_bonus) · еда/вода от нехватки 2×log1p (food/water_need_bonus).

### 6.3 Пайплайн награды
```
сырая награда (таблицы выше) → клип ±50 (env.cpp:1146)
→ rew_rms_.normalize_reward, клип ±10 (env.cpp:1303-1310, SB3-конвенция)
→ RolloutBuffer → GAE (gamma=0.999, lambda=0.98 по дефолту)
```
Настройка весов: дефолт — `configs/reward_v3.json` (подхватывается автоматически); кастомный — `train.py --reward-config <файл>` (можно частичный: отсутствующие ключи берутся из v3).

---

## 7. Curriculum (rl/curriculum.py — единый источник)

Стадия задаёт список разблокированных построек (вычисляется в Python, накопительно): **0 = все 32** · 1 = 14 базовых (дома/фермы/еда/вода/Refinery/Puerperal/Road…) · 2 = +11 (Sawmill/шахты/энергия…) · 3 = +7 (Big*/Atom/SuperHouse) = все 32. Этапы ≥3 эквивалентны 3 (пресеты покрывают все 32). Переключение: `--curriculum-schedule '200000:1,400000:2'`.

Ручной набор из вкладки «Курикулум» (`unlock_ids`) объединяется с пресетом этапа и **на этапе 0 становится единственным разрешённым списком** — например `--unlock-ids WaterChannel --curriculum-stage 0` оставляет только водоканал. Чекбокс «Ручной набор из Курикулума» (`use_curriculum_tab`) включает/выключает его; при выключенном работает только этап.

**Единая точка правды:** `rl/curriculum.py` (`build_state(stage, unlock_ids, use_curriculum_tab)` → `CurriculumState`), которой обязаны пользоваться ВСЕ среды — обучение (`rl/env_manager.py`), eval (`train_ui2/evaluator.run_eval`) и наблюдение (`watch_champion.py` + GUI-окно через `--curriculum JSON`). Python вычисляет разрешённый набор, C++ только хранит и применяет его (`ColonyEnvCpp::set_curriculum()` + `build_allowed()`); «нет ограничения» — это явный `all_builds=true`, а не пустой список. Запрет жёсткий: `Game::build` отказывает закрытым id до любых других проверок («Постройка закрыта курикулумом.»), GUI затемняет их иконки и не даёт выбрать, закрытое действие в `step` стоит ровно один `error_penalty` — как любая другая неудачная постройка. Сценарий записывается рядом с моделью (`curriculum_stage_at_best` / `unlock_ids` / `use_curriculum_tab` в `best_model.meta.json` и `meta.json`), оттуда его восстанавливают eval и «Наблюдение»; иначе модель, обученная на одном водоканале, строила бы прииск под полным набором зданий.

---

## 8. Обучение: параметры, чекпойнты, оценка

### 8.1 Ключевые гиперпараметры (rl/config.py)
| Параметр | Дефолт | Комментарий |
|---|---|---|
| n_envs / n_steps | 8 / 4096 | буфер = 32 768 шагов (flat ≈ 35 МБ VRAM) |
| batch_size / n_epochs | 8192 / 10 | 10 эпох — много; рекомендуется 4–6 + target_kl |
| learning_rate | 3e-4 | cosine decay до ~10% (LambdaLR в PPO) |
| gamma / gae_lambda | 0.999 / 0.98 | |
| ent_coef / vf_coef | 0.01 / 0.5 | |
| use_amp / amp_dtype | true / bfloat16 | для MLP 131K выгоды нет, можно off |
| eval_freq / eval_episodes | 100 000 / 20 | рекомендуется 500 000 / 10×3 сида |

**Формула VRAM буфера:** `n_envs × n_steps × (obs) × 4B`; minimap obs = 8×29×29 = 6728 float. ⚠ exp-конфиги с n_envs=1024 требуют 56+ ГБ — см. N12.

### 8.2 Файлы прогона (`~/colony_runs/models/<name>/`)
```
checkpoint_<steps>.pt          # модель + optimizer (+ .norm.json рядом)
final_model.pt / .norm.json
best_model.pt / .norm.json / .meta.json   # чемпион по composite score
_eval/_eval_temp.pt            # временный (удаляется)
normalization.json             # пишется при старте и каждом чекпойнте
```

### 8.3 Оценка и выбор чемпиона (rl/async_trainer.py::_eval → train_ui2/evaluator.py::run_eval)
* Политика грузится из чекпойнта (auto-detect MLP/CNN/Hybrid), **детерминированный argmax** с масками; нормализация obs загружается из файла и **замораживается** (`set_update(False)`).
* `episodes × max_days=10000`, сиды `seed+500000+ep`; метрики: days/people/bases/return (медиана по эпизодам и сидам).
* Score: `days×0.10 + bases×1.0 + people×0.10 + max(0,return)×1e-4 − штраф за дисперсию` (веса — `eval_score_weights`).
* Пороги: медиана `bases ≥ 5` (и p25 ≥ 70% от порога) и `days ≥ 730`; при улучшении score — сохранение best_model + meta. В конце обучения — турнир всех чекпойнтов.
* `auto_trainer.py` (Optuna) использует best_score как objective (fallback: best_reward).

---

## 9. UI-дашборд (train_ui2/)

`python run_train_ui2.py` → главное окно (вкладки Обучение/Мониторинг/Награды/Модели/Наблюдение). Обучение идёт в отдельном процессе `train_ui2/worker.py`, протокол — JSONL (protocol.py: progress/log/saved/done/error; команды boost_entropy / pause_training / resume_training / stop_training / reset_curriculum). Все параметры — ParamSpec из дефолтов rl/config.py (единый источник); награды редактируются на вкладке «Награды» (группы + абляции disable_*).

---

## 10. Тесты

```bash
python -m pytest                      # Все тесты (полный прогон требует torch + собранный colony_cpp.pyd)
python tests/test_gae.py            # GAE против референса
python tests/test_ppo_smoke.py      # PPO update без падений
python tests/test_evaluator.py      # run_eval end-to-end
python tests/test_reward_clip.py    # клип наград
python tests/test_milestones.py     # milestone-бонусы
python tests/test_curriculum.py     # стадии/маски
python tests/test_normalizer.py     # RunningMeanStd save/load
python tests/bench_per_step.py      # бенчмарк шага среды
# остальные файлы — по именам (tax, proximity, hybrid, minimap_radius, ui_full, ...)
```

---

## 11. Известные проблемы (историческая таблица старых аудитов — файлы PROJECT_AUDIT_FULL.md и др. в репозиторий не входят)

| ID | Кратко | Статус |
|---|---|---|
| B1 | `RewardConfig.from_dict` имел 5 дефолтов, отличных от датакласса/C++/JSON | подтверждён, фикс 1.1 |
| B2 | eval ищет `normalization.json`, который никто не пишет → eval без нормализации | подтверждён, фикс 1.4 |
| B6/B7 | approx_kl занижен ×12; LR-scheduler был no-op (нет decay) | подтверждены, фикс 1.3 |
| N1 | manual_tax_penalty: `rew -= (−0.5)` = бонус +0.5 | подтверждён, фикс 1.7 |
| N12 | exp-конфиги n_envs=1024 → 56 ГБ VRAM | подтверждён, фикс 2.3 |
| D1 | ~1 ГБ GPU→CPU копий на rollout | подтверждён, фикс 2.1 |
| F3/N7 | score на 76% из days; Optuna оптимизирует шум | подтверждены, фиксы 1.5+1.4 |
| N3/N4/N5 | хардкод-награды; счётчики «дней»=шаги; idle-ловушка при банкротстве | подтверждены, фиксы 3.7-3.9 |

Исправлены ранее: A1 (avg_return), A2 (is_hybrid), A3 (load_from_file), B5 (_last_terminateds). Корректно и проверено: GAE, сиды env'ов (base+i×10000), нормализация obs/reward в C++, маски DAY/WEEK, game_over_penalty (знак), 45 действий консистентны везде.

---

## 12. Быстрый индекс «где искать»

| Хочу… | Файл |
|---|---|
| поменять веса наград | configs/reward_v3.json → rl/config.py (датакласс) → include/colony/env.h (C++) — держать синхронно |
| добавить здание | configs/bases.json (+ порядок = индекс действия!) |
| понять, за что начислена награда | src/env.cpp:700-1150 (step), лог STEP с компонентами (:1112) |
| изменить наблюдение | src/env.cpp:461 (obs) / :651 (minimap) + пересборка |
| изменить гиперпараметры | rl/config.py (дефолты) / CLI train.py / configs/exp_*.json |
| логика eval/чемпиона | rl/async_trainer.py (_eval) + train_ui2/evaluator.py |
| маски действий | src/env.cpp:557 (action_mask), python/cpp_vecenv.py:184 |
| GAE/буфер | rl/rollout_buffer.py (compute_gae) |
| константы игры | include/colony/constants.h |
| сиды/детерминизм | python/cpp_vecenv.py:140, src/rng.cpp |
| почему обучение нестабильно | REPORT.md / REPORT_2026_09.md |
