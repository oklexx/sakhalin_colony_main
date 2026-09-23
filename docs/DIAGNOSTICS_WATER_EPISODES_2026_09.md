# WaterChannel / production-chain diagnostics (2026-09)

This note records a measured baseline before any Road-validator or action-mask
change. It also describes the terminal episode artifact added for real training
runs. The synthetic water-seeking rollouts below are **mechanics probes only**;
they are not evidence of what a learned checkpoint does.

## Water and Road baseline

The probe `tests/cpp/water_mask_check.cpp` was compiled against the current core
and run from the repository root. Its W0 checks reported:

- `WaterChannel.need_earth == LT_WATER` (`2`), price `18,310`.
- `Road.need_earth == LT_EVERYWHERE` (`10`), price `400`; the configured Road is
  therefore permitted on water if the shared placement/connectivity checks pass.

W1/W9 use an explicitly synthetic controller which chooses open directional
Road actions by direction-to-nearest-water and builds the WaterChannel as soon
as the action mask opens. Across seeds `1, 7, 21, 42, 100, 777, 31337, 2026`:

| Measure | Result |
|---|---:|
| WaterChannel mask vs. brute-force `Game::can_build_at` mismatches | 0 / 97 states |
| `find_lot()` missed / phantom legal sites | 0 / 0 |
| WaterChannel mask opened / channel built | 8 / 8 seeds |
| First-open step | 7–40 |
| Roads laid before stop | 89 |
| Roads placed on water / aggregate legal sites lost in this rollout | 0 / 0 |
| W9 worst roads-to-water / Euclidean-distance ratio | 2.70 |
| W8 masked-random WaterChannel builds | 0; 12 legal steps of 1,190 (1.01%) |

W8 is another policy-shaped probe, not a trained model: its diagnosis was
mostly insufficient funds (1,107 masked steps) and then no connected water lot
(71 steps). The W1/W9 rollout establishes that this mask/placement path can
reach water on those seeds; it does not establish that PPO learned to do so.

A second, targeted W6 mechanics probe is `tests/cpp/road_water_occupancy_check.cpp`.
It follows the same synthetic route only until a WaterChannel-legal water lot is
found, then deliberately calls `Game::build("Road", x, y)` on that exact lot.
This is not an agent action-selection experiment. Results:

| Seed | Roads before target | WC-legal sites before → after Road | Road legal/built | Target site removed |
|---:|---:|---:|:---:|:---:|
| 1 | 6 | 1 → 1 | yes / yes | yes |
| 7 | 16 | 1 → 0 | yes / yes | yes |
| 21 | 25 | 1 → 0 | yes / yes | yes |
| 42 | 31 | 1 → 3 | yes / yes | yes |
| 100 | 70 | 1 → 0 | yes / yes | yes |
| 777 | 77 | 2 → 2 | yes / yes | yes |
| 31337 | 83 | 1 → 3 | yes / yes | yes |
| 2026 | 89 | 1 → 1 | yes / yes | yes |

Thus Road **can** occupy the specific tile a WaterChannel could have used and
makes that tile unavailable (8/8). The total count of legal WaterChannel sites
fell on 3 seeds, stayed level on 3, and rose on 2: the new Road can also connect
other neighboring water tiles. In the baseline directional rollout, however,
none of the 89 placed roads landed on water. Keep both observations separate:
mechanics allow occupation; the tested heuristic happened not to choose it.

No Road validator, placement function, or action mask was changed. No trained
checkpoint is present in this checkout (only `tests/fake.pt`), so learned-policy
behavior still cannot be established here. Next compare terminal/step
diagnostics from actual checkpoints under fixed seeds; do not infer it from
these synthetic probes.

### Reproduction

From the repository root:

```bash
./scripts/cpp_checks.sh
```

The focused sources are `tests/cpp/water_mask_check.cpp` and
`tests/cpp/road_water_occupancy_check.cpp`; the check script compiles the complete
native core and runs both. Output paths from the exploratory one-off run were
`/tmp/water_mask_check` and `/tmp/road_water_probe` (not repository artifacts).

## Episode metrics and training JSONL

`EpisodeMetrics` now has explicit, stable diagnostic semantics:

- `chains_activated`: unique `(consumer build id, consumed resource)` pairs for
  which a working consumer and at least one working producer of that resource
  coexisted in a season during the episode. This is an active-link count, **not**
  a path depth.
- `max_chain_depth`: maximum number of distinct resource-conversion edges in a
  simple path in any one season's active resource-flow graph. Cycles cannot
  inflate depth. A measured synthetic `water → wood → food` fixture gives depth
  `2`; no active conversion gives `0`.
- `total_builds` and `builds_by_type`: successful in-step builds, including Road;
  `unique_build_types` is the count of distinct successfully built types.
- `reached_resources` and `reached_resource_ids`: count and canonical ids of
  resources ever produced by a working, finished building. The id list follows
  the resource index order (`gold`, `food`, `coal`, `iron`, `oil`, `stone`,
  `water`, `wood`, `energy`). `priority_reached` remains the reached subset with
  nonzero curriculum weight.
- `total_reward` is now a `double`, so fractional returns are not truncated.

At terminal auto-reset, native VecEnv places these values in
`info["episode"]["metrics"]`, alongside seed and final-state values. Python's
`CppVecEnv` JSON-decodes and preserves that info; `EnvManager` passes it to
`AsyncTrainer`. Training appends an independent, UTF-8, fsynced JSONL artifact:

```text
<model_dir>/episode_diagnostics.jsonl
```

The schema (`schema_version: 1`) has `run_start`, one `episode` row per terminal
environment episode, and `run_end` records. Episode rows include run id, seed,
training timestep, env index, curriculum stage, return/length, final days /
people / money / base count, and all `EpisodeMetrics`, including build counts
and resource ids. Missing metrics are recorded explicitly as
`metrics_available: false` rather than silently omitted. The worker records the
artifact filename in `meta.json` and logs its location. The file is append-only
and each row is flushed and `fsync`'d so it remains useful after an interrupted
run.

The extension handshake is now API version 7 and requires the
`episode_metrics` capability. Rebuild `colony_cpp` with `build_pyext.bat` before
training on Windows; a stale extension should fail fast instead of producing
empty chain diagnostics.

## Verification and known environment limits

- `./scripts/cpp_checks.sh` completed successfully after adding both
  `episode_metrics_check` and `road_water_occupancy_check`; the existing
  WaterChannel W0–W9 and regression checks also passed. The metrics fixture
  verified fractional return preservation, per-building counts,
  `chains_activated == 2`, `max_chain_depth == 2`, canonical reached resource
  ids, and terminal VecEnv JSON transport. The targeted Road probe reports
  `road_built=8/8` on a WaterChannel-legal cell and invalidates that exact target
  `8/8`; aggregate legal-site counts decrease on 3 seeds, stay level on 3, and
  increase on 2.
- Python JSONL / AsyncTrainer / VecEnv contracts are covered by
  `tests/test_episode_diagnostics.py`, `tests/test_async_trainer.py`, and
  `tests/test_cpp_vecenv_minimap.py`. Fine-tune preset and optimizer/scheduler
  regression tests were added as well.
- `python -m compileall -q rl train_ui2 watch_champion.py tests` and
  `git diff --check` pass. The JSONL writer and learning-rate helper also pass
  direct lightweight checks.
- Windows tree termination / UTF-8 tests are in `tests/test_watch_visual.py`;
  the POSIX process-tree helper and UTF-8 reconfiguration were exercised
  directly, but a real Windows run was not available.
- Full Python tests could not be run: this sandbox lacks `pytest`, `torch`,
  `PySide6`, and the compiled `colony_cpp` extension. No trained model
  checkpoint is present either, so actual PPO behavior on fixed seeds remains
  unmeasured; pybind11 linking and `.pyd` validation remain Windows-side checks.
