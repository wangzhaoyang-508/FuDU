"""Input and output helpers for detector-agnostic FuDU runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureTable:
    image_ids: list[str]
    features: np.ndarray
    labels: list[str] | None = None
    image_paths: list[str | None] | None = None


def read_feature_csv(path: str | Path, require_labels: bool = False) -> FeatureTable:
    """Read image features from CSV.

    Required column: ``image_id``.
    Optional columns: ``label``, ``image_path``.
    Feature columns are columns named ``f0``, ``f1``, ...; if absent, all
    non-metadata columns are treated as feature columns.
    """

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError("feature CSV must contain an image_id column")

        metadata = {"image_id", "label", "image_path"}
        feature_cols = [name for name in reader.fieldnames if name.startswith("f")]
        if not feature_cols:
            feature_cols = [name for name in reader.fieldnames if name not in metadata]
        if not feature_cols:
            raise ValueError("feature CSV must contain at least one feature column")

        image_ids: list[str] = []
        labels: list[str] = []
        image_paths: list[str | None] = []
        features: list[list[float]] = []
        for row in reader:
            image_ids.append(row["image_id"])
            labels.append(row.get("label", ""))
            image_paths.append(row.get("image_path") or None)
            features.append([float(row[col]) for col in feature_cols])

    if require_labels and not any(label != "" for label in labels):
        raise ValueError("feature CSV must contain label values for prototype building")

    return FeatureTable(
        image_ids=image_ids,
        features=np.asarray(features, dtype=np.float64),
        labels=labels if any(label != "" for label in labels) else None,
        image_paths=image_paths,
    )


def read_detection_jsonl(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read detector outputs from JSONL.

    Each line should be one of:

    ``{"image_id": "...", "boxes": [{...}, {...}]}``
    ``{"image_id": "...", "class_probs": [...], "loc_probs": [...]}``
    """

    detections: dict[str, list[dict[str, Any]]] = {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record.get("image_id")
            if not image_id:
                raise ValueError(f"missing image_id at {path}:{line_no}")
            boxes = record.get("boxes")
            if boxes is None:
                box = {k: v for k, v in record.items() if k != "image_id"}
                boxes = [box] if box else []
            detections[str(image_id)] = list(boxes)
    return detections


def write_scores_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write FuDU scoring records to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_id",
        "image_path",
        "global_uncertainty",
        "defect_uncertainty",
        "action",
        "sampling_probability",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_scores_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

