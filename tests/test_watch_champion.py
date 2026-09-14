"""Tests for watch_champion.py --curriculum-stage handling."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def test_curriculum_stage_arg_accepted():
    """--curriculum-stage should be accepted by the argument parser."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "watch_champion.py"), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "--curriculum-stage" in result.stdout


def test_curriculum_stage_from_meta(tmp_path):
    """When stage is None and meta exists, read stage from meta."""
    import json
    import watch_champion
    importlib.reload(watch_champion)

    meta_path = tmp_path / "best_model.meta.json"
    meta_path.write_text(json.dumps({"curriculum_stage_at_best": 3}))

    stage = watch_champion.read_stage_from_meta(tmp_path)

    assert stage == 3


def test_curriculum_stage_from_meta_missing(tmp_path):
    """When meta does not exist, read_stage_from_meta returns None."""
    import watch_champion
    importlib.reload(watch_champion)

    stage = watch_champion.read_stage_from_meta(tmp_path)

    assert stage is None


def test_visual_arg_accepted():
    """--visual should be accepted by the argument parser."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "watch_champion.py"), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "--visual" in result.stdout


def test_visual_launches_gui_exe(tmp_path):
    """When --visual is set, watch_champion launches the GUI exe."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        watch_champion.launch_visual_watch(
            model_dir=tmp_path,
            exe_path="test_gui.exe",
            actions_file=tmp_path / "actions.txt",
            state_file=tmp_path / "state.json",
            seed=42,
            map_size=200,
        )
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "test_gui.exe" in args
        assert "--headless-ai" in args
        assert "--actions-file" in args
        assert "--state-file" in args
        # PR 1: the curriculum flag is never skipped (explicit unrestricted).
        assert "--curriculum-all" in args


def test_write_action_file(tmp_path):
    """write_action should write the action int to the file."""
    import watch_champion
    importlib.reload(watch_champion)

    action_file = tmp_path / "actions.txt"
    watch_champion.write_action(action_file, 5)

    assert action_file.exists()
    assert int(action_file.read_text().strip()) == 5


def test_read_state_file(tmp_path):
    """read_state should parse the state JSON file including obs."""
    import json
    import watch_champion
    importlib.reload(watch_champion)

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "day": 10, "month": 3, "year": 2026,
        "people": 50, "bases": 5, "money": 1000,
        "action": 2, "terminated": False,
        "obs": [0.1, 0.2, 0.3, 0.4, 0.5],
    }))

    state = watch_champion.read_state(state_file)
    assert state["day"] == 10
    assert state["people"] == 50
    assert state["terminated"] is False
    assert len(state["obs"]) == 5
    assert state["obs"][0] == 0.1


def test_read_state_missing_file(tmp_path):
    """read_state should return None if file doesn't exist."""
    import watch_champion
    importlib.reload(watch_champion)

    state = watch_champion.read_state(tmp_path / "nonexistent.json")
    assert state is None


def test_visual_launch_passes_curriculum_json(tmp_path):
    """The GUI exe must receive the computed state over --curriculum JSON."""
    import json
    import watch_champion
    importlib.reload(watch_champion)
    from rl.curriculum import build_state

    mock_proc = MagicMock()
    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        watch_champion.launch_visual_watch(
            model_dir=tmp_path,
            exe_path="test_gui.exe",
            actions_file=tmp_path / "actions.txt",
            state_file=tmp_path / "state.json",
            seed=42,
            map_size=200,
            curriculum=build_state(0, "WaterChannel", True),
        )
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "--curriculum" in args
        payload = json.loads(args[args.index("--curriculum") + 1])
        assert payload["all_builds"] is False
        assert payload["allowed_builds"] == ["WaterChannel"]
        assert "--stage" not in args and "--unlock-ids" not in args
