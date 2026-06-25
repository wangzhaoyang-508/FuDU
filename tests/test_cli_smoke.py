import tempfile
import unittest
from pathlib import Path

from fudu.cli import main
from fudu.io import read_scores_csv


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTest(unittest.TestCase):
    def test_demo_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prototypes = tmp_path / "prototypes.npz"
            scores = tmp_path / "scores.csv"
            selected = tmp_path / "selected.txt"

            self.assertEqual(
                main(
                    [
                        "build-prototypes",
                        "--features",
                        str(ROOT / "examples" / "initial_features.csv"),
                        "--output",
                        str(prototypes),
                        "--n-normal",
                        "2",
                        "--n-defect",
                        "2",
                    ]
                ),
                0,
            )
            self.assertTrue(prototypes.exists())

            self.assertEqual(
                main(
                    [
                        "score",
                        "--features",
                        str(ROOT / "examples" / "stream_features.csv"),
                        "--detections",
                        str(ROOT / "examples" / "detections.jsonl"),
                        "--prototypes",
                        str(prototypes),
                        "--output",
                        str(scores),
                        "--seed",
                        "0",
                    ]
                ),
                0,
            )
            rows = read_scores_csv(scores)
            self.assertEqual(len(rows), 4)
            self.assertIn("MS", {row["action"] for row in rows})

            self.assertEqual(
                main(["make-yolo-list", "--scores", str(scores), "--output", str(selected)]),
                0,
            )
            self.assertTrue(selected.exists())


if __name__ == "__main__":
    unittest.main()

