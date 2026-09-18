# Mechanic curriculum gate (2026-09)

## Contract

The action space remains fixed at 49 slots. Only the applicability mask changes:

| mechanic allow-list item | fixed manager slots |
|---|---|
| `improve_land` | `IMPROVE_LAND` |
| `preservation` | `PRESERVE`, `UNPRESERVE` |
| `credit` | `TAKE_LOAN`, `REPAY_LOAN` |

`CurriculumState.enabled_mechanics` is the canonical Python allow-list. It is
transported through `set_curriculum()` and returned by `curriculum()` for parity
checks. Missing metadata/transport fields retain legacy all-enabled behavior;
new `Config()` runs use the early-economy defaults.

The default new-run schedule is additive:

```json
{
  "disabled_mechanics": ["improve_land", "preservation", "credit"],
  "mechanics_unlock_schedule": [
    [200000, ["improve_land", "preservation"]],
    [500000, ["credit"]]
  ]
}
```

Entries are additions, thresholds must be sorted, and C++ refuses a mid-run
re-lock. A schedule therefore cannot silently remove preservation from a colony
that already has preserved buildings.

## Enforcement and policy

Python masks are not a security boundary. C++ `ColonyEnvCpp::step()` rejects a
direct/stale call to a disabled manager action with `error_penalty`, increments
only the interaction diagnostic, and returns before `advance_day()`, tax
settlement, building placement, or money/credit mutation. This is tested against
all five fixed slots. Enabling a mechanic restores the normal applicability
predicate; it does not make an otherwise inapplicable action legal.

The critic receives the rollout action mask through a zero-initialized,
value-only mask projection. This preserves observation width and actor output
width while giving the value function the same availability context at rollout,
PPO update, and bootstrap. Old checkpoints load with the projection missing
(strict=False); their actor/critic heads and 49-logit contract are unchanged.

## Automatic tax-to-debt is separate

`credit` gates only manual `TAKE_LOAN` and `REPAY_LOAN`. Automatic tax settlement
is controlled independently by `Config.tax_to_debt` / the C++ constructor flag:

- `true` (RL default): an unpaid annual/main tax is paid from cash and the
  remainder is converted to bank debt; the calendar continues. This does not
  count as the policy choosing `TAKE_LOAN`.
- `false` (dialogue/GUI mode): the old explicit tax-failure behavior is retained;
  no hidden manual loan is created.

The setting is persisted in checkpoint metadata and restored by evaluation and
watch paths. Manual credit can be disabled while automatic tax-to-debt remains
on; the eventual debt-limit/game-over signal is then the explicit consequence.

## Compatibility paths

- `Config.curriculum_state(step)` computes buildings, resources, observation
  version, and mechanic allow-list in one object.
- `EnvManager.set_curriculum_progress()` applies additive mechanic unlocks
  without changing `n_actions`, action order, or model output heads.
- Async trainer metadata stores both the declarative disabled/schedule config
  and the snapshot `enabled_mechanics` at each checkpoint.
- `resolve_state()` restores the snapshot; if only declarative metadata is
  available, it reconstructs the snapshot at the stored checkpoint step.
- Evaluator, watch, single-env, vector-env, and visual JSON paths all use the
  same transport field and parity checks.

## Verification

The relevant checks are:

```text
python -m py_compile rl/*.py python/cpp_env.py python/cpp_vecenv.py train_ui2/evaluator.py watch_champion.py
 g++ -std=c++17 -Iinclude -Iinclude/third_party -fsyntax-only src/env.cpp
g++ -std=c++17 -Iinclude -Iinclude/third_party -fsyntax-only tests/cpp/curriculum_check.cpp
```

The C++ curriculum harness verifies initial zero masks, no calendar/economy
mutation on direct disabled calls, exact applicability masks after unlock, and
JSON transport. Python contract tests verify monotonic scheduling and
checkpoint round-trip behavior.
