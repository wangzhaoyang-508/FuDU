"""Streaming FuDU scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fuzzy import FuzzyDecision, FuzzySampler
from .io import FeatureTable
from .prototypes import PrototypeLibrary
from .uncertainty import image_defect_uncertainty


@dataclass(frozen=True)
class ScoreRecord:
    image_id: str
    image_path: str | None
    global_uncertainty: float
    defect_uncertainty: float
    decision: FuzzyDecision
    selected: bool

    def to_row(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path or "",
            "global_uncertainty": round(self.global_uncertainty, 6),
            "defect_uncertainty": round(self.defect_uncertainty, 6),
            "action": self.decision.action,
            "sampling_probability": self.decision.probability,
            "selected": int(self.selected),
        }


def score_records(
    features: FeatureTable,
    detections: dict[str, list[dict[str, Any]]],
    prototypes: PrototypeLibrary,
    sampler: FuzzySampler | None = None,
    seed: int = 0,
    stochastic: bool = True,
    deterministic_threshold: float = 0.5,
    deue_kwargs: dict[str, Any] | None = None,
) -> list[ScoreRecord]:
    """Score a stream of images and decide which images to annotate."""

    sampler = sampler or FuzzySampler()
    rng = np.random.default_rng(seed)
    deue_kwargs = deue_kwargs or {}
    paths = features.image_paths or [None] * len(features.image_ids)
    ug_values = prototypes.global_uncertainty(features.features)

    records: list[ScoreRecord] = []
    for idx, image_id in enumerate(features.image_ids):
        ug = float(ug_values[idx])
        ud = image_defect_uncertainty(detections.get(image_id, []), **deue_kwargs)
        decision = sampler.evaluate(ug, ud)
        if stochastic:
            selected = bool(rng.random() < decision.probability)
        else:
            selected = bool(decision.probability >= deterministic_threshold)
        records.append(
            ScoreRecord(
                image_id=image_id,
                image_path=paths[idx],
                global_uncertainty=ug,
                defect_uncertainty=ud,
                decision=decision,
                selected=selected,
            )
        )
    return records

