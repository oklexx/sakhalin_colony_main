"""Behavioral-cloning dataset loader for ``--record-demo`` JSONL logs.

See docs/IMITATION_LEARNING_2026_09.md for the full design write-up. In
short: ``src/gui.cpp --record-demo PATH`` appends one JSON object per RL
timestep while a human plays, routed through the exact same
``env.step(action)`` call the RL rollout loop uses (see
``env_step_and_record`` in gui.cpp). Each line carries::

    {"action": int, "reward": float, "terminated": bool,
     "obs": [float, ...], "action_mask": [float, ...]}

This module turns one or more such files into a flat, shuffle-able dataset
of (obs, action, action_mask) transitions for supervised pretraining
(``rl/bc_pretrain.py``), plus a helper to compute matching normalization
stats so the resulting checkpoint's observation scale lines up with what
``train.py --resume-model`` expects.
"""
from __future__ import annotations

import glob
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# obs_size -> obs_version. Mirrors rl.curriculum._OBS_SIZE_BY_VERSION (kept
# private there to steer callers toward its compat-check helpers); duplicated
# here as a tiny literal because this module only ever needs the reverse
# direction (obs_size -> version) and importing the private name would be
# more fragile than a 3-entry dict.
_OBS_SIZE_TO_VERSION = {248: 0, 289: 1, 299: 2}


def infer_obs_version(obs_size: int) -> int:
    """Map a demo file's flat obs width to the obs_version it was recorded with.

    Fails fast (matching rl/curriculum.py's obs-compat checks) on an
    unrecognized width instead of silently guessing: a wrong obs_version tag
    would make ``train.py --resume-model`` believe the checkpoint expects a
    different observation layout than the one it actually saw.
    """
    try:
        return _OBS_SIZE_TO_VERSION[int(obs_size)]
    except KeyError:
        raise ValueError(
            f"unrecognized obs_size={obs_size} in demo file: expected one of "
            f"{sorted(_OBS_SIZE_TO_VERSION)} (see rl/curriculum.py "
            "_OBS_SIZE_BY_VERSION). Pass --obs-version explicitly if this "
            "demo really does come from a non-standard build."
        ) from None


@dataclass
class DemoStep:
    """One recorded (obs, action) decision, in the same shape env.step() uses."""

    obs: np.ndarray          # float32 [obs_size]
    action: int
    action_mask: np.ndarray  # float32 [n_actions], 1.0 = available
    reward: float
    terminated: bool
    episode: int              # 0-based episode index, unique across all loaded files
    source: str                # originating file path (for error messages)


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: malformed demo JSONL line: {e}") from e


_REQUIRED_FIELDS = ("action", "reward", "terminated", "obs", "action_mask")


def load_demo_file(path: str | Path) -> list[DemoStep]:
    """Parse one ``--record-demo`` JSONL file into a list of `DemoStep`.

    Episodes are delimited by ``terminated: true`` lines (see
    ``demo_record_step`` in src/gui.cpp). A trailing, non-terminated run at
    end-of-file — recording simply stopped mid-episode, e.g. the human quit
    the GUI without the colony collapsing — is kept as its own (open)
    episode rather than discarded: every recorded decision is still a valid
    (obs, action) supervision pair regardless of how the episode ended.
    """
    path = Path(path)
    steps: list[DemoStep] = []
    episode = 0
    for lineno, row in _iter_jsonl(path):
        missing = [f for f in _REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"{path}:{lineno}: demo line missing field(s) {missing}")
        steps.append(DemoStep(
            obs=np.asarray(row["obs"], dtype=np.float32),
            action=int(row["action"]),
            action_mask=np.asarray(row["action_mask"], dtype=np.float32),
            reward=float(row["reward"]),
            terminated=bool(row["terminated"]),
            episode=episode,
            source=str(path),
        ))
        if row["terminated"]:
            episode += 1
    return steps


def load_demo_files(paths: Sequence[str | Path]) -> list[DemoStep]:
    """Load and concatenate multiple demo files.

    Episode indices are renumbered globally (offset per file) so a
    train/val split by episode across several recorded sessions doesn't
    collide episode 0 of file A with episode 0 of file B.
    """
    all_steps: list[DemoStep] = []
    offset = 0
    for p in paths:
        file_steps = load_demo_file(p)
        if not file_steps:
            continue
        for s in file_steps:
            s.episode += offset
        offset = file_steps[-1].episode + 1
        all_steps.extend(file_steps)
    return all_steps


def expand_demo_paths(patterns: list[str]) -> list[str]:
    """Expand CLI ``--demo`` args: directories glob for ``*.jsonl``, files/globs pass through."""
    out: list[str] = []
    for p in patterns:
        path = Path(p)
        if path.is_dir():
            found = sorted(str(f) for f in path.glob("*.jsonl"))
            if not found:
                raise ValueError(f"no *.jsonl demo files found in directory: {p}")
            out.extend(found)
        else:
            matches = sorted(glob.glob(p))
            if not matches:
                raise ValueError(f"no files match --demo pattern: {p}")
            out.extend(matches)
    return out


def validate_consistent_dims(steps: list[DemoStep]) -> tuple[int, int]:
    """Ensure every step shares the same obs_size / n_actions; return them.

    Mixing demo files recorded against different colony_cpp builds (an
    obs_version bump, or a ``configs/bases.json`` edit that changed the
    build count) would silently misalign features instead of crashing — this
    fails fast, matching rl/curriculum.py's own obs/version compatibility
    checks.
    """
    if not steps:
        raise ValueError("no demo steps loaded (empty dataset)")
    obs_size = int(steps[0].obs.shape[0])
    n_actions = int(steps[0].action_mask.shape[0])
    for s in steps:
        if int(s.obs.shape[0]) != obs_size:
            raise ValueError(
                f"{s.source}: obs_size {s.obs.shape[0]} != {obs_size} "
                "(mixing demo files recorded with different obs_version builds?)"
            )
        if int(s.action_mask.shape[0]) != n_actions:
            raise ValueError(
                f"{s.source}: n_actions {s.action_mask.shape[0]} != {n_actions} "
                "(mixing demo files recorded with different colony_cpp builds?)"
            )
        if not (0 <= s.action < n_actions):
            raise ValueError(f"{s.source}: action {s.action} out of range [0, {n_actions})")
    return obs_size, n_actions


def compute_normalization(steps: list[DemoStep], clip: float = 10.0) -> dict:
    """Mean/var/count over demo observations.

    Flat stats (``mean``, ``var``, ``count``, ``obs_size``, ``clip``) — the
    shape of ``python/cpp_env.py::Normalizer.to_dict()``. Do NOT write this
    dict straight to disk for ``train.py --resume-model``: the vec-env
    loader (``ColonyVecEnvCpp::load_normalization``) wants the nested
    ``obs_rms``/``rew_rms`` schema — wrap with
    ``rl.bc_pretrain.vec_normalization_payload`` first (pre-2026-09-29 this
    file was written flat and PPO resume crashed/mis-normalized on it).

    Note: this is a plain batch mean/var over all recorded frames, not an
    online Welford update seeded with the engine's count=1 pseudo-count (see
    include/colony/running_mean_std.h). For a dataset of any real size the
    difference is negligible; it only means a *subsequent* PPO fine-tuning
    run's first few live updates blend in a hair faster than if the exact
    seed count had matched.
    """
    obs = np.stack([s.obs for s in steps], axis=0).astype(np.float64)
    mean = obs.mean(axis=0)
    var = obs.var(axis=0)
    return {
        "mean": mean.tolist(),
        "var": var.tolist(),
        "count": int(obs.shape[0]),
        "obs_size": int(obs.shape[1]),
        "clip": float(clip),
    }


def split_by_episode(
    steps: list[DemoStep], val_frac: float, seed: int = 0
) -> tuple[list[DemoStep], list[DemoStep]]:
    """Split into (train, val) by episode, not by transition.

    Splitting by individual transition would leak near-duplicate consecutive
    frames of the same episode across train/val, making the validation loss
    an overly optimistic estimate of how the policy generalizes to unseen
    play.
    """
    episodes = sorted({s.episode for s in steps})
    if val_frac <= 0 or len(episodes) < 2:
        return steps, []
    rng = np.random.default_rng(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val_ids = set(shuffled[:n_val])
    train = [s for s in steps if s.episode not in val_ids]
    val = [s for s in steps if s.episode in val_ids]
    if not train:
        # Degenerate case (e.g. a single giant episode landed entirely in
        # val): fall back to no split rather than training on nothing.
        return steps, []
    return train, val


class BCDataset(Dataset):
    """torch Dataset of (obs, action, action_mask) demo transitions."""

    def __init__(self, steps: list[DemoStep]):
        self.steps = steps

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.steps[idx]
        return (
            torch.from_numpy(s.obs),
            torch.tensor(s.action, dtype=torch.long),
            torch.from_numpy(s.action_mask),
        )
