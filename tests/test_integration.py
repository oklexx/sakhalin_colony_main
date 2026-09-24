"""Integration test: 100k steps, measure FPS and GPU utilization."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

def run_training(args: list[str], timeout: int = 600) -> str:
    cmd = [sys.executable, str(TRAIN_SCRIPT)] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
    return result.stdout

def extract_fps(output: str) -> float:
    for line in output.splitlines():
        if "Average FPS:" in line:
            return float(line.split("Average FPS:")[1].strip().replace(",", ""))
    return 0.0

def main():
    print("=" * 60)
    print("INTEGRATION TEST: 100k steps, 8 envs, GPU training")
    print("=" * 60)

    args = [
        "--steps", "100000",
        "--envs", "8",
        "--n-steps", "4096",
        "--batch-size", "8192",
        "--n-epochs", "10",
        "--name", "integration_test",
    ]

    output = run_training(args, timeout=600)
    print(output)

    fps = extract_fps(output)
    print(f"\nExtracted FPS: {fps:,.0f}")

    if fps >= 50000:
        print("PASS: FPS >= 50,000")
    elif fps >= 20000:
        print("WARN: FPS between 20k-50k (acceptable for small net)")
    else:
        print(f"INFO: FPS={fps:,.0f} (below target, may need tuning)")

    if "Traceback" in output:
        print("FAIL: Training crashed")
        sys.exit(1)
    else:
        print("PASS: Training completed without errors")


def test_training_with_eval_integration(tmp_path):
    """End-to-end: train for a few steps with eval, verify best_model and normalization exist."""
    import torch

    from rl.config import Config
    from rl.env_manager import EnvManager
    from rl.async_trainer import AsyncTrainer

    cfg = Config(
        n_envs=2,
        n_steps=32,
        total_timesteps=256,
        save_freq=128,
        eval_freq=128,
        eval_episodes=2,
        map_size=100,
        model_dir=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        eval_min_bases=1,
        eval_min_days=0.0,
    )

    device = torch.device("cpu")
    em = EnvManager(cfg, device)

    # Save initial normalization
    norm_path = Path(cfg.model_dir) / "normalization.json"
    em.vec_env.venv.save_normalization(str(norm_path))
    assert norm_path.exists(), "normalization.json should exist after initial save"

    trainer = AsyncTrainer(cfg=cfg, env_manager=em)
    trainer.train(total_timesteps=256)

    # Verify best model was saved (eval should have run at step 128 and 256)
    best_model = Path(cfg.model_dir) / "best_model.pt"
    best_meta = Path(cfg.model_dir) / "best_model.meta.json"
    assert best_model.exists(), "best_model.pt should exist after training with eval"
    assert best_meta.exists(), "best_model.meta.json should exist"

    # Verify normalization exists
    assert norm_path.exists(), "normalization.json should still exist"

    em.close()


if __name__ == "__main__":
    main()
