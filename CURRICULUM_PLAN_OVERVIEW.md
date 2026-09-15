# Сводный обзор: план Curriculum (PR 3 → 1 → 2 → 6 → 4 → 5)

База: `32cbf78`. Порядок слияния — `PR3 → PR1 → PR2 → PR6 → PR4 → PR5`
(ветка `pr5/obs-v1`, HEAD `a93c629`). Источник истины по скоупу —
`uploads/CURRICULUM_IMPLEMENTATION_PLAN.md` (§1–§8).

## Сводная таблица

| PR | Коммит | Тема | Файлов | +/− | Сьют | Harness |
|----|--------|------|--------|-----|------|---------|
| 3 | `a8af0ab` | Handshake stale-бинарника | 22 | +419/−25 | — | — |
| 1 | `a09c28a` | Единый контракт курикулума | 22 | +927/−330 | 239 ✓ | — |
| 2 | `21e2654` | Жёсткий гейт на уровне Game | 8 | +163/−24 | 239 ✓ | 27/27 |
| 6 | `05ff93c` | Консистентность наград/масок | 8 | +224/−24 | 243 ✓ | 37/37 |
| 4 | `2f11f09` | Ресурсный курикулум (веса) | 21 | +492/−29 | 255 ✓ | 52/52 |
| 5 | `a93c629` | obs v1 — рамка (287) | 22 | +681/−23 | 275 ✓ | 66/66 |

Итого: ~103 file-touches, +2906/−455. Везде «✓» = passed, 2 skipped;
`tests/test_smoke_fixes.py` исключён везде (сломан на базе, до плана).

---

## PR 3 — Handshake (`a8af0ab`)

**Проблема:** stale `colony_cpp.pyd` в git тихо затенял исправленные C++
исходники — клонировавший получал старое поведение при корректных сорцах.

- `extension_info()` в расширении: version / features / src_sha; CMake
  вшивает SHA коммита, единый вывод `python/` для всех конфигураций.
- `python/colony_cpp_api.py`: `require_colony()` бросает
  `StaleExtensionError` вместо warning; escape hatch `--allow-stale-pyd` /
  `COLONY_ALLOW_STALE_PYD=1`. Handshake в train.py, evaluator, watch,
  `CppVecEnv`/`CppColonyEnv`.
- Неизвестные ключи `RewardConfig` — ошибка, не warning.
- Гигиена: бинарники untracked (`.pyd/.so/.exp/.lib/.exe`,
  `python/Release/`); `build_pyext.bat` без захардкоженных путей + self-check.
- Тесты: `tests/test_extension_handshake.py` (на fake-модулях, без torch).

## PR 1 — Единый контракт (`a09c28a`)

**Ядро плана:** Python — единственное место вычисления разрешённого набора
(`CurriculumState` в `rl/curriculum.py`), C++ — только хранение и применение
(`set_curriculum()` + `build_allowed()`). «Нет ограничения» — явный
`all_builds=true`, никогда побочный эффект пустого списка.

- C++: удалены `CURRICULUM_STAGE_*`-пресеты, `rebuild_unlocked`, split
  stage/manual; `Curriculum::from_json` для `--curriculum JSON` с fail-fast;
  `stage_report` для obs-фичи и дампов; extension v2, фичи
  `set_curriculum`/`curriculum`; `--curriculum`/`--curriculum-all` в main и GUI.
- Python: `curriculum=` у `CppColonyEnv`/`CppVecEnv`/`make_*` (duck-typed,
  без torch); `Config.curriculum_state()`; EnvManager строит ОДНО состояние,
  проверяет read-back parity на старте, расписание переприменяет одно
  состояние; evaluator/watch через `resolve_state()`; GUI-watch — всегда
  `--curriculum JSON`.
- Тесты: 23 contract + 6 parity (согласие 4 путей создания среды);
  мигрированы scope/handshake/reward/tax/watch-сьюты. Сьют: 239 ✓.

## PR 2 — Жёсткий гейт (`21e2654`)

`Game::build` отказывает закрытым id первым, до любых других проверок
(«Постройка закрыта курикулумом.»). Гейт пересаживается в
ctor/`set_curriculum`/`reset` (reset пересоздаёт Game из gateless-копии —
без пересадки гейт тихо исчезал бы после первого reset); песочница
`colony_cpp.Game` по умолчанию без гейта; контекст гейта переживает
undo/snapshot.

- `env.step` BUILD: закрытые действия — ранний выход через nullopt-cell,
  стоят ровно ОДИН `error_penalty` (раньше двойной). Маска остаётся
  (GUI ходит в `Game::build` напрямую).
- UI: затемнение закрытых иконок + отказ с сообщением; `do_build_area`
  хранит первую ошибку; консольное меню печатает `[locked]`.
  `ui/main_window.py` пропущен: мёртвый код (нет импортеров, `core.*`
  отсутствует в репо).
- Harness +10 (27/27). Сьют: 239 ✓.

## PR 6 — Консистентность наград/масок (`05ff93c`)

1. Закрытый BUILD — чистый отказ: ровно `error_penalty`, ранний return —
   без `advance_day`, дневных/каталоговых бонусов, obs не меняется. Раньше
   день проходил и капал `tax_daily_bonus`, т.е. заблокированная попытка
   могла дать +0.30 при `error_penalty=0` (= DAY). Бухгалтерия консистентна,
   счётчик tax-grace заморожен для отказов.
2. Унифицированный mask-fill `-1e9` в eval/watch (был `float('-inf')`):
   робастность к полностью закрытой маске, нет NaN в top-3 логах. Как в PPO.
3. Warning вырожденного сценария: на reset печатаются причины по каждому
   разрешённому зданию (`price > money` / нет лота), если доступно 0/32
   BUILD-действий. Замер из плана воспроизводится точь-в-точь
   (seed 7, 82k, WaterChannel-only → лот).
- Harness 37/37 (twin-тест фиксирует намеренную асимметрию:
  gated = чистый штраф без advance vs money-fail = день проходит).
  Сьют: 243 ✓.

## PR 4 — Ресурсный курикулум (`2f11f09`)

`curriculum_resources` записывался/хранился, но никогда не читался.
Теперь течёт `build_state → C++ set_curriculum → compute_catalog`, где ОДИН
множитель масштабирует `extract_weight_` (`all_resources=true` — legacy
бит-в-бит; смена по расписанию сама пересчитывает награду).

- Семантика: 1.0 выбранным / 0.0 остальным; all/empty ⇒ все 1.0; полный
  набор из 9 нормализуется к `all_resources` (зеркало правила зданий).
  Неизвестные имена — громкая ошибка (`parse_resources`).
- Транспорты: dict/`from_json` (fail-fast на неверном размере), фича
  расширения `resource_curriculum` (теперь required в handshake).
- C++: множитель + опциональный `priority_count_over_allowed` (дефолт false);
  трекинг `extracted_` отвязан от бонусов (reached = ВСЕ достигнутые);
  метрика `priority_reached` + биндинги; `dump_obs pr=water,wood`;
  канон `Sunduk::resource_name`; веса float→double (точная parity).
- Python: parse/build_state/resolve/meta/eval/watch, `train.py
  --curriculum-resources`, лог воркера `priority=water,wood (weights=[...])`;
  parity EnvManager проверяет оба измерения всегда (старый early-out
  пропускал веса при `all_builds`).
- Тесты: harness 52/52 (twin autumn-прогоны, ручной расчёт), сьют 255 ✓.
- Честная фиксация «план vs реальность»: вода/золото заблокированы
  экономикой, а не весами (замер −9.7 на скриптованном 250-дневном эпизоде);
  `w_food = w_gold = 0` всегда (ноль зданий-потребителей).

## PR 5 — obs v1 (`a93c629`)

Модель видит рамку: 287 = 246 + 9 весов + 32 бита; v0 — строгий префикс v1.
`counts`/`idle_by_type` не фильтруются никогда, `n_actions` = 45 не тронут.

- C++: `obs_size()` +`SUNDUK_SIZE`+`n_build_` при `obs_version>=1`; tail
  строго после v0-хвоста; эффективные веса (та же формула, что экономика) +
  биты `build_allowed`; `set_curriculum` бросает при смене версии;
  `obs_mask_locked_catalog` (дефолт off, только каталог); транспорт версии
  во from_json/from_dict/to_dict.
- Python: `Config.obs_version=1` (дефолт v1 везде, включая None-путь);
  версия explicit-only (из meta НЕ восстанавливается — stored v0 ОШИБКА);
  персистентность в ckpt/meta/norm; `Normalizer.load` отклоняет чужие
  размеры; `obs mismatch`-ошибки в eval/watch/resume (тег + ширина тензоров);
  `--obs-version` в train/watch/eval; parity покрывает версию (stale .so
  читается как v0 и падает громко); алиас `--priority-resources`.
- По ходу найден и починен настоящий баг: у `AsyncTrainer` свои
  `_curriculum_meta`/`_curriculum_kwargs` мимо `Config.curriculum_meta()` —
  eval-мета писалась без версии, in-training eval падал бы посреди прогона.
- Тесты: harness +14 (66/66), +10 contract, +5 parity, +1 normalizer,
  +3 evaluator; сьют 275 ✓. Смоук §7 проверен (v1+v0 прогоны, watch,
  resume-mismatch). README §5.2/§7 обновлены.

---

## Что осталось открытым (не в скоупе плана, зафиксировано)

1. **Windows GUI/main compile** через `build_gui.bat` — проверить перед
   мержем (отмечено в PR 1 и PR 2, Linux-сборка зелёная).
2. **`train.py` не пишет run `meta.json`** (пишет только worker/GUI-путь) —
   предсуществующий пробел: у CLI-прогонов нет мета для restore/eval.
   PR 5 это не чинил (policy-width чек покрывает bare `.pt`).
3. **Замер экономики §7** (Δ`avg_return`/entropy/`top_actions` на 200k,
   до/после) — следующий шаг, см. ниже.
4. Нормалайзер и константные фичи: при фиксированной стадии рамка
   константна и после RunningMeanStd стремится к ~0 — архитектура по плану,
   информативна при расписании стадий; отдельно не исследовалось.
5. `ui/main_window.py` — мёртвый код (см. PR 2), кандидат на удаление.
