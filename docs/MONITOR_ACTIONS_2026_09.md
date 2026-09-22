# Мониторинг: «водоканал появился, процент рос — и строка пропала»

`docs/MONITOR_ACTIONS_2026_09.md` · 2026-09-23

## 1. Симптом

Вкладка **Мониторинг**, панель «Строительство зданий»: строка «Водоканал»
появляется, её процент подрастает за несколько обновлений, затем строка
исчезает. Читалось как «мониторинг теряет данные» / «вода пропала из сценария».

## 2. Причина (три независимых механизма, все — в способе измерения)

| # | Механизм | Где |
|---|---|---|
| M1 | Доли считались по `deque(maxlen=1000)`, а роллаут = `n_steps * n_envs` = 32768 (дефолт). Панель видела ~3% роллаута: 0 нажатий в хвосте = «действие пропало» | `rl/async_trainer.py` (`_action_history`) |
| M2 | Отсечка `[:15]` в тренере + стабильная сортировка: при равных `count` побеждал меньший **индекс** действия, и здание с `count=1` вылетало из списка, как только набиралось 15 действий с `count≥2` — при том что его доля не падала | `rl/async_trainer.py::_top_actions_dict` |
| M3 | BUILD-действие легально, только если оно открыто курикулумом, `money ≥ price` и `find_lot()` нашёл свободный участок (для WaterChannel — `LT_WATER`, к которому доехала дорога). Окно легальности короткое, а монитор о нём молчал | `src/action_mask.cpp:40-57`, `tests/cpp/water_mask_check.cpp` W1–W3/W6 |

M3 — не баг, а свойство сценария: см. `include/colony/constants.h` («the
WaterChannel action was masked on 0/2000 random steps») и
`docs/BALANCE_WATERCHANNEL_2026_09.md`: из 457 состояний маска закрыта
**деньгами** в 240 и **отсутствием клетки** в 210. Построил водоканал →
`18 310` ушло → маска закрылась → следующий роллаут без строк. Это и есть
нормальное поведение: «появился → подрос → пропал» = «вода стала доступна →
строим → окно закрылось».

Попутно найдено и закрыто:

- **подпись «курикулум: этап N (0%)» врела всегда**: `AsyncTrainer` звал
  `self.em.get_curriculum_progress(...)`, которого не существовало ни в
  `EnvManager`, ни в vec env, ни в биндингах; `except Exception` глушил
  `AttributeError`, и UI получал `0.0` на каждом прогоне;
- `curriculum_available_actions` считался в трене, ехал через `worker.py` и
  `protocol.py`, но в окне не показывался;
- классификация «это постройка или нет» в UI сравнивала `uk in ALL_BUILD_IDS`
  регистрозависимо (`WATERCHANNEL` ≠ `WaterChannel`) — «голые» id уезжали в
  панель «Активности»;
- счётчик и история циклов сидели в одном `try/except: pass` — молча
  обнулённый мониторинг было не отличить от «политика ничего не строит».

## 3. Контракт после правки

1. **Окно = ровно один роллаут.** `AsyncTrainer._record_rollout_actions` копит
   счётчик по всем `n_steps * n_envs` действиям, `_reset_rollout_actions`
   обнуляет его перед сбором. Доля читается буквально: «сколько шагов
   роллаута ушло на действие». Скользящее окно осталось как явный
   отладочный режим: `_calculate_action_distribution(window=N)`.
2. **Никакого top-N в данных.** `Config.monitor_action_top = 0` по умолчанию —
   в UI едут все действия с `count > 0` плюс сырые `action_counts`.
   `train_ui2/monitor.py::rows_with_sticky` решает, сколько строк влезает по
   высоте панели, и закрепляет `KEY_ACTIONS_MONITOR` (водоканал, дорога,
   ферма, сад, дом): нулевая закреплённая строка остаётся видимой серой —
   это диагноз, а не пустое место.
3. **Легальность отдельной шкалой.** `rl/action_monitor.py::ActionLegalityMonitor`
   в `EnvManager.collect_step` суммирует ту же маску, из которой сэмплируется
   действие; `pop_action_legality()` отдаёт долю шагов (0..100) за роллаут.
   UI рисует её серым «следом» под полоской и подписью `0.4% · легально 12%`.
   Отсюда вердикт (`monitor.action_verdict`): `заблокировано маской` (<1%),
   `окно легальности узкое` (<25%), `легально, но не выбирает` (доля 0 при
   легальности ≥25%), иначе `выбирает`. Легальность неизвестна → вердикта нет
   (врать запрещено). Флаг `Config.monitor_action_legality` (по умолчанию вкл).
4. **Прогресс этапа — из одного источника.** `rl/curriculum.py::stage_progress`
   считает `(step - start) / (next - start)` по `curriculum_schedule`;
   если следующего порога нет (этап последний или расписания нет) —
   `mode="fixed"`, `progress=None`, и UI пишет «прогресс не измеряется», а не
   вечный ноль. Недоступная метрика логируется **один раз**
   (`AsyncTrainer._monitor_warn_once`), а не глушится.

## 4. Что осталось за кадром (осознанно)

- **Атрибуция причины маски** (деньги / нет клетки / курикулум) живёт в C++
  (`water_mask_check` W2) и в UI не переносится: для неё нужен новый биндинг и
  rebuild `colony_cpp` (handshake), а цель правки — снять неоднозначность
  «нельзя vs не хочет» малой ценой. Следующий шаг, если потребуется:
  `action_mask_reason_counts()` в `src/bindings.cpp` → та же колонка в панели.
- `loop_detector` по-прежнему смотрит последние 1000 действий: у детектора
  своя семантика (недавний цикл), расширение окна сделало бы «циклы»
  срабатывающими на весь роллаут.
- Eval (`train_ui2/evaluator.py`) считает дни/базы, а не доли действий;
  легальность в eval не измеряется (там argmax без накопления статистики).

## 5. Файлы и тесты

- `rl/action_monitor.py` (новый) — `ActionLegalityMonitor`.
- `rl/curriculum.py` — `stage_progress()`.
- `rl/env_manager.py` — `_action_legality` в `collect_step`,
  `pop_action_legality()`, `get_curriculum_progress()`.
- `rl/async_trainer.py` — счётчики роллаута, `_track_step_actions`,
  `_top_actions_dict(limit)`, `_curriculum_progress_view`, поля `TrainMetrics`
  (`action_counts`, `action_legality`, `action_total_steps`,
  `curriculum_progress_valid`).
- `rl/config.py` — `monitor_action_top`, `monitor_action_legality`.
- `train_ui2/monitor.py` (новый, без Qt) — `split_by_panel`, `rows_with_sticky`,
  `action_verdict`, `watch_report`.
- `train_ui2/charts.py` — `Bars(keep=, legend=)` + `set_items(items, legality)`.
- `train_ui2/constants.py` — `KEY_ACTIONS_MONITOR`.
- `train_ui2/protocol.py`, `worker.py`, `main_window.py` — проводка полей и
  подпись «ключевые действия».
- Тесты: `tests/test_monitor_action_stats.py` (25), обновлённые
  `tests/test_async_trainer.py::test_top_actions_dict_omits_zeros_and_maps_names`
  и `tests/test_ui2_smoke.py` (панель теперь 1 + 5 закрепленных строк).
