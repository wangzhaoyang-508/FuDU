"""Run the tiny FuDU demo without installing the package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fudu.cli import main


def run() -> None:
    out = ROOT / "runs" / "demo"
    main(
        [
            "build-prototypes",
            "--features",
            str(ROOT / "examples" / "initial_features.csv"),
            "--output",
            str(out / "prototypes.npz"),
            "--n-normal",
            "2",
            "--n-defect",
            "2",
            "--alpha",
            "4",
            "--beta",
            "1",
        ]
    )
    main(
        [
            "score",
            "--features",
            str(ROOT / "examples" / "stream_features.csv"),
            "--detections",
            str(ROOT / "examples" / "detections.jsonl"),
            "--prototypes",
            str(out / "prototypes.npz"),
            "--output",
            str(out / "scores.csv"),
            "--fuzzy-preset",
            "nuclear_fuel_rod",
            "--seed",
            "0",
        ]
    )
    main(
        [
            "make-yolo-list",
            "--scores",
            str(out / "scores.csv"),
            "--output",
            str(out / "selected_train.txt"),
        ]
    )


if __name__ == "__main__":
    run()
