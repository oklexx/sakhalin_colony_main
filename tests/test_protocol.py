import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_ui2 import protocol as P


def test_roundtrip_ready():
    msg = P.decode(P.encode(P.ReadyMsg()))
    assert isinstance(msg, P.ReadyMsg)


def test_roundtrip_log():
    m = P.LogMsg(level="warn", message="пробел  и  табы\t")
    d = P.decode(P.encode(m))
    assert isinstance(d, P.LogMsg)
    assert d.level == "warn"
    assert d.message == "пробел  и  табы\t"


def test_log_invalid_level():
    with pytest.raises(ValueError):
        P.LogMsg(level="debug", message="x")


def test_roundtrip_progress():
    m = P.ProgressMsg(done=1, total=10, fps=1.5, best_reward=2.25,
                      episodes=3, policy_loss=0.1, value_loss=0.2,
                      entropy=0.3, kl=0.001)
    d = P.decode(P.encode(m))
    assert d.done == 1
    assert d.total == 10
    assert d.fps == 1.5
    assert d.best_reward == 2.25
    assert d.episodes == 3
    assert d.kl == 0.001


def test_roundtrip_saved():
    d = P.decode(P.encode(P.SavedMsg(path="C:/models/a.pt")))
    assert isinstance(d, P.SavedMsg)
    assert d.path == "C:/models/a.pt"


def test_roundtrip_done():
    m = P.DoneMsg(total=100, time_s=5.5, best_reward=9.9, episodes=4)
    d = P.decode(P.encode(m))
    assert d.total == 100
    assert d.time_s == 5.5
    assert d.episodes == 4


def test_roundtrip_error():
    d = P.decode(P.encode(P.ErrorMsg(message="boom")))
    assert isinstance(d, P.ErrorMsg)
    assert d.message == "boom"


def test_decode_broken_json():
    with pytest.raises(ValueError):
        P.decode("{not json")


def test_decode_unknown_type():
    with pytest.raises(ValueError):
        P.decode(json.dumps({"type": "wat"}))


def test_decode_missing_fields():
    with pytest.raises(ValueError):
        P.decode(json.dumps({"type": "progress", "done": 1}))


def test_decode_non_object():
    with pytest.raises(ValueError):
        P.decode("[1, 2, 3]")


def test_decode_empty_line():
    with pytest.raises(ValueError):
        P.decode("   ")


def test_encode_no_trailing_newline():
    assert not P.encode(P.ReadyMsg()).endswith("\n")


def test_encode_nan_sanitized():
    d = P.decode(P.encode(P.ProgressMsg(done=0, total=1, fps=float("nan"))))
    assert d.fps == 0.0


def test_encode_stop():
    d = json.loads(P.encode_stop())
    assert d["type"] == "command"
    # С 2026-09-24 канонное имя одно на UI/воркер/трейнер (P1-4 ревью);
    # короткое "stop" принимается только как устаревший алиас.
    assert d["cmd"] == P.CMD_STOP == "stop_training"
    assert d["payload"]["final_save"] is True
