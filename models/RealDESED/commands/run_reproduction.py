import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
DATA_DIR = ROOT_DIR / "data"


def run_cmd(cmd, env=None, check=True):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"\n[RUNNING] {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=ROOT_DIR, env=full_env)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        sys.exit(res.returncode)
    return res.returncode


def step_compute_annotator_scores():
    print("\n" + "=" * 50)
    print("Step: Computing Annotator Quality Scores")
    print("=" * 50)
    cmd = [
        str(VENV_PYTHON),
        str(ROOT_DIR / "compute_annotator_score.py"),
        "--root", str(DATA_DIR),
    ]
    run_cmd(cmd)


def step_reproduce_baseline(
    batch_size=12,
    accumulate_grad=4,
    n_epochs=1,
    num_workers=0,
    check_val_every_n_epoch=None,
    experiment_name="atst_f_reproduction",
):
    print("\n" + "=" * 50)
    print("Step: Running RealDESED Baseline Training")
    print("=" * 50)
    env = {
        "WANDB_MODE": "offline",
        "PYTHONPATH": str(ROOT_DIR),
    }

    if check_val_every_n_epoch is None:
        check_val_every_n_epoch = min(5, max(1, n_epochs))

    cmd = [
        str(VENV_PYTHON),
        str(ROOT_DIR / "train.py"),
        "--dataset_path", str(DATA_DIR),
        "--output_dir", str(ROOT_DIR / "runs"),
        "--experiment_name", experiment_name,
        "--batch_size", str(batch_size),
        "--accumulate_grad_batches", str(accumulate_grad),
        "--n_epochs", str(n_epochs),
        "--num_workers", str(num_workers),
        "--precision", "16",
        "--check_val_every_n_epoch", str(check_val_every_n_epoch),
        "--train_annotation_aggregation", "Weighted Soft",
        "--annotator_alpha", "16",
        "--include_reviewed_train_files",
    ]
    run_cmd(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_annotator_scores", action="store_true", help="Skip compute_annotator_score.py")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size per step (12 recommended for 4GB VRAM)")
    parser.add_argument("--accumulate_grad", type=int, default=4, help="Gradient accumulation steps (12*4 = 48)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None, help="Check validation every N epochs")
    parser.add_argument("--experiment_name", type=str, default="atst_f_reproduction")
    args = parser.parse_args()

    if not args.skip_annotator_scores:
        step_compute_annotator_scores()

    if not args.skip_training:
        step_reproduce_baseline(
            batch_size=args.batch_size,
            accumulate_grad=args.accumulate_grad,
            n_epochs=args.n_epochs,
            num_workers=args.num_workers,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            experiment_name=args.experiment_name,
        )

    print("\nReproduction finished successfully!")


if __name__ == "__main__":
    main()
