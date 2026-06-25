"""Dual-entropy defect uncertainty evaluator (DeUE)."""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12


def classification_entropy(class_probs: Any, normalize: bool = True) -> float:
    """Return classification entropy, normalized by log(C) by default."""

    probs = _normalize_distribution(np.asarray(class_probs, dtype=np.float64))
    if probs.ndim != 1:
        raise ValueError("class_probs must be a 1D probability vector")
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, EPS))))
    if normalize and len(probs) > 1:
        entropy /= float(np.log(len(probs)))
    return float(np.clip(entropy, 0.0, 1.0 if normalize else np.inf))


def localization_entropy(loc_probs: Any, normalize: bool = True) -> float:
    """Return mean localization-bin entropy for one box.

    ``loc_probs`` may have shape ``(4, K)`` for left/top/right/bottom bins, or
    simply ``(K,)`` if a detector exposes one localization distribution.
    """

    probs = np.asarray(loc_probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if probs.ndim != 2:
        raise ValueError("loc_probs must be a 1D or 2D array")

    rows = np.vstack([_normalize_distribution(row) for row in probs])
    entropy = -np.sum(rows * np.log(np.maximum(rows, EPS)), axis=1)
    value = float(np.mean(entropy))
    if normalize and rows.shape[1] > 1:
        value /= float(np.log(rows.shape[1]))
    return float(np.clip(value, 0.0, 1.0 if normalize else np.inf))


def box_uncertainty(
    class_probs: Any | None = None,
    confidence: float | None = None,
    loc_probs: Any | None = None,
    class_entropy_norm: float | None = None,
    loc_entropy_norm: float | None = None,
    w_cls: float = 1.0,
    w_loc: float = 2.0,
    divisor: str | float = "paper",
    clip: bool = True,
) -> float:
    """Compute FuDU box-level defect uncertainty.

    The paper writes ``1/3 * (w1 H_cls + w2 H_loc + 1 - conf)`` and adopts
    ``w1=1, w2=2``. With these weights the raw value may exceed 1, so this
    implementation clips to [0, 1] by default. Pass ``divisor="sum_weights"``
    to divide by ``w_cls + w_loc + 1`` instead.
    """

    if class_entropy_norm is None:
        if class_probs is None:
            class_entropy_norm = 0.0
        else:
            class_entropy_norm = classification_entropy(class_probs, normalize=True)

    if confidence is None:
        if class_probs is None:
            confidence = 0.0
        else:
            confidence = float(np.max(_normalize_distribution(np.asarray(class_probs, dtype=np.float64))))

    if loc_entropy_norm is None:
        loc_entropy_norm = 0.0 if loc_probs is None else localization_entropy(loc_probs, normalize=True)

    if divisor == "paper":
        denom = 3.0
    elif divisor == "sum_weights":
        denom = float(w_cls + w_loc + 1.0)
    else:
        denom = float(divisor)
    if denom <= 0:
        raise ValueError("divisor must be positive")

    value = (float(w_cls) * float(class_entropy_norm) + float(w_loc) * float(loc_entropy_norm) + 1.0 - float(confidence)) / denom
    if clip:
        value = float(np.clip(value, 0.0, 1.0))
    return value


def image_defect_uncertainty(
    boxes: list[dict[str, Any]] | None,
    w_cls: float = 1.0,
    w_loc: float = 2.0,
    divisor: str | float = "paper",
) -> float:
    """Aggregate box-level uncertainty to image-level Ud with max pooling."""

    if not boxes:
        return 0.0
    values = []
    for box in boxes:
        values.append(
            box_uncertainty(
                class_probs=box.get("class_probs"),
                confidence=box.get("confidence", box.get("conf")),
                loc_probs=box.get("loc_probs"),
                class_entropy_norm=box.get("class_entropy_norm", box.get("h_cls")),
                loc_entropy_norm=box.get("loc_entropy_norm", box.get("h_loc")),
                w_cls=w_cls,
                w_loc=w_loc,
                divisor=divisor,
            )
        )
    return float(max(values))


def _normalize_distribution(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("distribution must be 1D")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= EPS:
        return np.full_like(arr, 1.0 / len(arr), dtype=np.float64)
    return arr / total

