import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_ui2 import worker as W


def test_build_parser_train_mode():
    p = W._build_parser()
    args = p.parse_args(["--config", "c.json", "--name", "run1", "--output", "out.jsonl"])
    assert args.config == "c.json"
    assert args.name == "run1"
    assert args.output == "out.jsonl"
    assert args.eval_model is None


def test_build_parser_eval_mode():
    p = W._build_parser()
    args = p.parse_args(["--eval-model", "m.pt", "--episodes", "3", "--output", "out.jsonl"])
    assert args.eval_model == "m.pt"
    assert args.episodes == 3
    assert args.config is None


def test_build_parser_requires_mode():
    p = W._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_read_config():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"total_timesteps": 100, "n_envs": 2}, f)
        path = f.name
    try:
        d = W._read_config(Path(path))
        assert d["total_timesteps"] == 100
        assert d["n_envs"] == 2
    finally:
        os.unlink(path)


def test_read_config_bad_json():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write("{broken")
        path = f.name
    try:
        with pytest.raises(json.JSONDecodeError):
            W._read_config(Path(path))
    finally:
        os.unlink(path)


def test_msgfile_writes_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        mf = W.MsgFile(path)
        mf.open()
        mf.write(W.P.ReadyMsg())
        mf.write(W.P.LogMsg(level="info", message="hello"))
        mf.write(W.P.ProgressMsg(done=1, total=10, fps=5.0))
        mf.close()
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 3
        d0 = json.loads(lines[0])
        assert d0["type"] == "ready"
        d1 = json.loads(lines[1])
        assert d1["type"] == "log"
        assert d1["message"] == "hello"
        d2 = json.loads(lines[2])
        assert d2["type"] == "progress"
        assert d2["done"] == 1
    finally:
        os.unlink(path)


def test_msgfile_thread_safety():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        mf = W.MsgFile(path)
        mf.open()
        n_threads = 4
        n_msgs = 25

        def worker(i):
            for j in range(n_msgs):
                mf.write(W.P.LogMsg(level="info", message=f"t{i}-{j}"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        mf.close()
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == n_threads * n_msgs
        for line in lines:
            json.loads(line)
    finally:
        os.unlink(path)


def test_msgfile_write_raw():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        mf = W.MsgFile(path)
        mf.open()
        mf.write_raw('{"type":"eval_result","days":100}\n')
        mf.close()
        with open(path, encoding="utf-8") as f:
            d = json.loads(f.readline().strip())
        assert d["type"] == "eval_result"
        assert d["days"] == 100
    finally:
        os.unlink(path)


def test_watch_stdin_stop(monkeypatch):
    import io
    stop = threading.Event()
    fake_stdin = io.StringIO('{"cmd":"stop"}\n')
    monkeypatch.setattr(W.sys, "stdin", fake_stdin)
    W._watch_stdin(stop)
    assert stop.is_set()


def test_watch_stdin_ignores_bad_json(monkeypatch):
    import io
    stop = threading.Event()
    fake_stdin = io.StringIO("not json\n")
    monkeypatch.setattr(W.sys, "stdin", fake_stdin)
    W._watch_stdin(stop)
    assert not stop.is_set()
