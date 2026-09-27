# Imitation learning from recorded human games (2026-09)

Status: implemented, v1. Companion to `docs/CHAMPION_SELECTION_2026_09.md`
(same review/upgrade pass). This document is the canonical reference for
"обучение модели на уже сыгранной игре" — recording a human GUI session and
pretraining an RL-compatible policy from it.

## 1. Problem statement

`src/gui.cpp` lets a human play the same colony simulation the RL agent
trains on (`build 2>&1 | ... colony_gui`), but historically every click went
straight to `Game`/`GameManager` methods (`g.build()`, `gm.restore()`, ...)
and never touched `ColonyEnvCpp::step()`. That means no `(obs, action,
reward, action_mask)` tuple was ever produced for a human decision — there
was nothing a behavioral-cloning (BC) pipeline could train on.

The ask: let a human play a game, save that game, and warm-start (or purely
pretrain) an RL policy from it.

## 2. Design decision: route recording through `env.step()`, not through raw clicks

The RL action space (`env.n_actions()` = 50 for the current 33-building
`configs/bases.json`) is **always an auto-targeting/auto-amount space**:
every action ID resolves a concrete cell, target building, or resource
amount *inside* `env.step()` — no action ever carries an explicit (x, y) or
a custom quantity. Concretely (see `src/env.cpp`):

* `BUILD_<id>` picks a cell via `find_lot()` (BFS from existing buildings).
* `MGR:repair` / `MGR:destroy` operate on `find_slowest_base()`.
* `MGR:preserve` picks the least-worn unpreserved building; `MGR:unpreserve`
  releases the first preserved one.
* `MGR:sell` sells everything above a fixed buffer (200 units/resource);
  `MGR:buy_food` tops food up to a fixed level (400); `MGR:credit_take`/
  `MGR:credit_give` move a fixed 50000; `MGR:manual_tax` pays tax with a
  fixed penalty.

Because of this, a human's mouse click and an RL action ID are compatible at
exactly one level of abstraction: *which decision*, not *which cell/amount*.
Recording therefore means intercepting the human's decision **before** it
reaches `Game`/`GameManager` directly, and instead calling
`env.step(action_id)` — the exact call the RL rollout loop makes — so the
env's own auto-targeting resolves the concrete effect, and the resulting
`(obs_before, action, reward, terminated, action_mask_before)` is logged
exactly as a rollout would have produced it.

This is *not* a lossy workaround; it's the only representation an RL policy
can actually be trained to imitate. A human who clicked a specific tile
still has that tile choice discarded (the RL agent could never reproduce
it), but the *decision itself* (e.g. "repair something now") is captured
faithfully.

## 3. What gets recorded, opt-in via `--record-demo PATH`

`colony_gui --record-demo demos/session1.jsonl [other GUI flags]` opens
`PATH` in append mode and, for the rest of the session, routes every
recordable human decision through a new `env_step_and_record(env, action)`
helper (`src/gui.cpp`) instead of the normal direct `Game`/`GameManager`
call. Without the flag, nothing changes — every existing code path is
untouched byte-for-byte.

`env_step_and_record`:
1. snapshots `env.obs()` and `env.action_mask()` **before** stepping (this
   is the state the decision was made from — the same thing an RL rollout
   would see);
2. calls `env.step(action)` (identical call PPO rollouts make);
3. appends one JSONL line: `demo_record_step(pre_obs, action, pre_mask,
   reward, terminated)`;
4. returns the `StepOut` so the caller can still update `game_over` etc.

### 3.1 Action coverage

| GUI action (toolbar icon / hotkey) | RL action | Recorded? |
|---|---|---|
| Build (palette click + drag, or popup icon), `do_build_area()` | `BUILD_<id>` (one `env.step()` call **per placed building**, not per cell-area) | ✅ |
| Улучшить землю (`G`) | `MGR:improve_land` | ✅ |
| Починить (`R`) | `MGR:repair` | ✅ |
| Починить всё (`A`) | `MGR:restore_all` | ✅ |
| Снести (`D`) | `MGR:destroy` | ✅ |
| Консервация toggle (`P`) | `MGR:preserve` **or** `MGR:unpreserve` — resolved by checking the clicked building's `preserved` flag *before* toggling, so an "unpreserve" click is never mislabeled as "preserve" | ✅ |
| День (Space/Enter/`N`) | `A_DAY` | ✅ |
| Неделя (`W`) | `A_WEEK` | ✅ |
| Найти слабейшее (camera jump) | *(read-only, no env action exists for it)* | n/a — doesn't mutate state |
| Отмена / Undo (`U`) | *(no RL equivalent — PPO/BC never undoes a step)* | ❌ **disabled** while recording (shows a status message) instead of silently producing an untracked state rewind |
| Рынок: купить/продать (`B`/`S`, dialog) | `MGR:sell` / `MGR:buy_food` exist, but with **fixed** heuristic amounts; the GUI dialog lets the human type an arbitrary per-resource quantity | ❌ **dialog itself is blocked** while recording (see §3.2) |
| Банк: взять/вернуть (`K`, dialog) | `MGR:credit_take` / `MGR:credit_give` exist, fixed at 50000; the dialog accepts an arbitrary typed amount | ❌ **dialog itself is blocked** while recording |
| — | `MGR:manual_tax` | ❌ no GUI affordance in this build at all (not a recording gap — there is no button/hotkey for it) |
| — | `ROAD_E/W/S/N` (4 directional road actions) | ❌ no GUI affordance in this build at all |

### 3.2 Why Market/Bank dialogs are blocked outright, not just "unrecorded"

`draw_market`/`draw_bank` call `g.market_buy()/market_sell()/bank_take()/
bank_give()` **directly**, bypassing `env.step()` entirely, with whatever
quantity the human typed. If recording just ignored these calls, the demo
file would silently miss a real state mutation between two logged steps —
the next recorded `obs` would already reflect money/resources the log never
explained, which would corrupt BC training data (the model would see
unexplained jumps and learn wrong obs→action associations). Instead, while
`--record-demo` is active, opening these dialogs (`B`/`S`/`K` or their
toolbar icons) is refused with a status message, forcing the session to stay
within the RL-representable action set. This is intentionally more
restrictive than "just play normally" — a documented v1 trade-off, not a
bug. Supporting free-amount market/bank actions in a future BC-only branch
(a policy head with a continuous or discretized-amount output) is listed
under Future Work (§7).

### 3.3 A pacing side effect that must be understood before recording

`ColonyEnvCpp::step()` advances the game by **one full day for every
action except `A_WEEK`** (see `src/env.cpp`: every branch falls through to
`g.advance_day()`). In *normal* (non-recording) GUI play, clicking a manager
button (repair, preserve, ...) is "free" — only the День/Неделя buttons
advance time. In `--record-demo` mode, because every recorded decision goes
through `env.step()`, **every recorded click also advances one day**,
exactly like an RL agent's turn. This is required for the data to be valid
BC training material (temporal alignment with rollout semantics — the
policy is trained on "what to do this turn", one action per turn), but it
means recorded sessions play at a different pace than normal GUI sessions:
a human who wants to repair three buildings and then advance a day will
actually advance four days while recording. This is called out in-game too
(status bar messages use "(демо-режим)" suffixes) and must be explained to
anyone recording a demo.

### 3.4 Demo file schema

Plain JSONL, one line per recorded step:

```json
{"action": 7, "reward": 0.42, "terminated": false, "obs": [...], "action_mask": [...]}
```

Field names deliberately mirror `ai_write_state`'s existing
`obs`/`action_mask`/`action`/`reward`/`terminated` fields (the headless-AI
IPC protocol already used by `--headless-ai`), so the same JSON shape reused
across two parts of this codebase is one less thing to keep in sync. There
is no separate meta file — a demo file is self-describing: `len(obs)`
determines `obs_version` (`rl/bc_dataset.py::infer_obs_version`, mirroring
`rl/curriculum.py::_OBS_SIZE_BY_VERSION`), and `len(action_mask)`
determines `n_actions`.

Episodes are delimited by `terminated: true` lines. A trailing run with no
final `terminated: true` (recording stopped mid-episode, e.g. the human
just closed the GUI) is kept as its own episode — every recorded decision
is still valid training data regardless of how the file ends.

`--record-demo` and `--headless-ai` are mutually exclusive: headless mode
already has its own step-logging protocol (state.json/action.json IPC), and
combining the two would double-drive `env.step()` from two different
sources. If both are passed, `--record-demo` is ignored with a `WARNING` on
stderr.

## 4. BC pretraining pipeline

### 4.1 `rl/bc_dataset.py`

Pure numpy/json, no dependency on the compiled `colony_cpp` extension (a
practical requirement in environments — like this sandbox — where
`colony_cpp` cannot be built). Key pieces:

* `load_demo_file` / `load_demo_files` — parse one or more `--record-demo`
  JSONL files into a flat list of `DemoStep` (obs, action, action_mask,
  reward, terminated, episode id, source path). Episode indices are
  renumbered globally across files so a later split-by-episode doesn't
  collide file A's episode 0 with file B's episode 0.
* `expand_demo_paths` — `--demo` accepts explicit files, glob patterns, or
  directories (globbed for `*.jsonl`).
* `validate_consistent_dims` — fails fast if demo files were recorded
  against different `obs_version`/`n_build` builds (mixing them would
  silently misalign features), matching the fail-fast philosophy already
  used by `rl/curriculum.py`'s obs-compat checks.
* `compute_normalization` — batch mean/var/count in the exact schema
  `python/cpp_env.py::Normalizer.to_dict()` uses (`mean`/`var`/`count`/
  `obs_size`/`clip`), so it can be written straight to a
  `normalization.json`.
* `split_by_episode` — train/val split **by episode**, not by transition,
  to avoid leaking near-duplicate consecutive frames of the same episode
  across the split.
* `BCDataset` — a `torch.utils.data.Dataset` over `(obs, action,
  action_mask)`.

### 4.2 `rl/bc_pretrain.py`

```
python -m rl.bc_pretrain --demo demos/session1.jsonl demos/session2.jsonl \
    --out models/bc_pretrained/model.pt --epochs 30
# or:
python -m rl.bc_pretrain --demo demos/ --out models/bc_pretrained/model.pt
```

What it does:

1. Loads and validates the demo file(s), infers `obs_size`/`n_actions`
   (and `obs_version`, overridable with `--obs-version`).
2. Computes normalization stats over all recorded observations and
   normalizes with the **exact** formula
   `include/colony/running_mean_std.h::RunningMeanStd::normalize` uses
   (`(x - mean) / sqrt(|var| + 1e-8)`, NaN/Inf → 0, clipped to ±10) — see
   `rl.bc_pretrain.normalize_obs`. Using a different convention here would
   make the pretrained weights actively wrong once resumed into PPO
   training against the C++ env's own running normalizer.
3. Builds `rl.actor_critic.ActorCritic(obs_size, n_actions, hidden_sizes,
   device)` — the same flat-obs MLP `rl/ppo.py`/`train.py` use by default.
   (Demo files only carry the flat `obs`, not the 32×32 minimap, so BC
   pretraining only targets the MLP policy in v1 — see §7.)
4. Trains with masked cross-entropy: `logits = model(obs,
   mask)[0].masked_fill(mask == 0, -1e9)` then
   `cross_entropy(logits, recorded_action)` — mirroring exactly how
   `rl/ppo.py::PPO.collect_step`/`evaluate` mask invalid actions before
   sampling, so a resumed policy's masked-logit behavior matches what it
   was trained on.
5. Tracks train/val loss and top-1 accuracy per epoch, keeps the
   best-val-loss (or best-train-loss, if there's no val split) weights, and
   supports early stopping (`--patience`, default 8 epochs).
6. Saves a checkpoint with the **exact schema `rl/ppo.py::PPO.save()`
   writes**: `model_state`, `optimizer_state`, `buffer_pos`, `n_actions`,
   `hidden_sizes`, `obs_version`, `obs_size`. This is what makes the
   checkpoint a drop-in artifact rather than a special case:
   * `train_ui2/evaluator.py::_load_policy` infers architecture purely from
     tensor shapes in `model_state` (plus the `obs_size`/`n_actions`/
     `hidden_sizes` hints) — it loads a BC checkpoint with zero new code.
   * `python train.py --resume-model <checkpoint>` warm-starts PPO from it,
     after checking `obs_version` compatibility
     (`rl.curriculum.check_obs_version_compat`) and the flat input width
     (`check_policy_obs_compat`).
7. Writes `normalization.json` **and** `<checkpoint>.norm.json` next to the
   checkpoint (same content, two names) because `train.py --resume-model`
   looks for exactly one of
   `Path(resume_model).with_suffix(".norm.json")` or
   `Path(resume_model).parent / "normalization.json"` — and otherwise
   prints `WARNING: normalization file not found` and fine-tunes on
   **unnormalized** observations, silently ruining the run.

### 4.3 Resuming into PPO

```
python train.py --name my_run --resume-model models/bc_pretrained/model.pt \
    --resume-lr 1e-4 ...
```

No new code needed: `train.py`'s existing `--resume-model` path
(lines ~239-292) already handles loading `model_state`, validating
`obs_version`/obs-width compatibility, restoring (or, on failure, silently
skipping) the optimizer state, and loading the sibling normalization file.
A BC checkpoint slots into that path unmodified.

## 5. Tests

* `tests/test_bc_dataset.py` — JSONL parsing (including episode
  segmentation on `terminated`, malformed-line/missing-field errors,
  multi-file episode renumbering), path expansion, dimension-consistency
  fail-fast checks, normalization math, and episode-level train/val
  splitting. Pure numpy/json — runs without `colony_cpp`.
* `tests/test_bc_pretrain.py` — the `normalize_obs` formula (including the
  NaN/Inf → 0 and clip behavior), and an end-to-end BC training run on a
  small synthetic-but-learnable demo (`action = argmax(obs[:n_actions])`)
  that asserts accuracy actually rises above chance, the checkpoint has the
  exact expected keys, and — critically — that the checkpoint loads through
  `train_ui2/evaluator.py::_load_policy` unmodified (the real compatibility
  contract, not just a raw `torch.load`).

Both suites run in this repo's sandbox despite `colony_cpp` being
unbuildable here, because neither depends on the compiled extension.

## 6. Files touched/added

* `src/gui.cpp` — `#include <fstream>`; demo-recording infra
  (`g_record_demo`, `g_demo_file`, `demo_record_open/step/close`,
  `env_step_and_record`); `--record-demo PATH` CLI flag; `MGR_*` action-id
  constants in `main()`; toolbar-click **and** keyboard-shortcut handlers
  for build/improve/repair/repair-all/destroy/preserve/day/week now branch
  on `g_record_demo`; Undo and Market/Bank dialogs are refused with a
  status message while recording.
* `rl/bc_dataset.py` (new) — demo JSONL loader.
* `rl/bc_pretrain.py` (new) — BC training script.
* `tests/test_bc_dataset.py`, `tests/test_bc_pretrain.py` (new).

## 7. Explicit scope-outs / future work

* **Market/Bank free-amount actions.** Would need either (a) a
  discretized-amount action space extension (e.g. sell 25%/50%/100% of
  surplus) that the RL env itself would also have to support, or (b) a
  BC-only continuous "amount" head that has no PPO counterpart. Rejected
  for v1: it would either change the shared RL action space (out of scope
  for a recording feature) or produce a checkpoint architecture PPO can't
  resume.
* **Minimap/CNN and hybrid policies.** The demo file only carries the flat
  `obs` (matching what `ai_write_state` already logs); a minimap-aware BC
  path would need `env.minimap()` captured per recorded step too. Not
  implemented — `rl/bc_pretrain.py` only targets `ActorCritic` (flat MLP).
* **Reward-weighted / filtered BC.** v1 trains plain per-step cross-entropy
  cloning; it does not down-weight or discard low-reward human decisions
  (e.g. a human accidentally demolishing something valuable is cloned just
  as confidently as a good decision). A reward- or return-weighted BC loss
  is a natural follow-up once there's a larger corpus of recorded games.
* **`MGR:manual_tax` and the 4 `ROAD_E/W/S/N` actions** have no GUI
  affordance in this build at all — not a recording limitation, a gap in
  the GUI's own feature set relative to the RL action space. They can never
  appear in a demo file until the GUI grows a control for them.
