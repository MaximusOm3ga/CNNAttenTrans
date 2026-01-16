import subprocess
import sys
from pathlib import Path


def run_step(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    print(f"\n=== Running {script} ===", flush=True)
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        cwd=str(script.parent),
    )
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({result.returncode}): {script}")


def main() -> None:
    base = Path(__file__).resolve().parent

    steps = [
        base / "modelCNN" / "training.py",
        base / "modelCNN" / "token_ex.py",
        base / "transformerModel" / "training.py",
        base / "transformerModel" / "eval_unseen.py",
    ]

    for step in steps:
        run_step(step)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
