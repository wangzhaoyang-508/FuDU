"""Prototype-based global uncertainty quantification (PGUQ).

This module implements the detector-agnostic part of FuDU:

    Ug = sigmoid(alpha * min(d_normal, d_defect) - beta)

where distances are measured from the current image feature to the nearest
normal and defect prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

EPS = 1e-12


@dataclass
class PrototypeLibrary:
    """Normal and defect feature prototypes for PGUQ."""

    normal: np.ndarray
    defect: np.ndarray
    alpha: float = 1.0
    beta: float = 0.0
    normalize: bool = False

    def __post_init__(self) -> None:
        self.normal = _as_2d_float(self.normal, "normal")
        self.defect = _as_2d_float(self.defect, "defect")
        if self.normal.shape[1] != self.defect.shape[1]:
            raise ValueError("normal and defect prototypes must share feature dimension")
        if self.normalize:
            self.normal = _l2_normalize(self.normal)
            self.defect = _l2_normalize(self.defect)

    @property
    def feature_dim(self) -> int:
        return int(self.normal.shape[1])

    def nearest_distances(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return distances to nearest normal, defect, and either prototype set."""

        feats = _as_features(features, self.feature_dim, self.normalize)
        dn = _nearest_l2(feats, self.normal)
        dd = _nearest_l2(feats, self.defect)
        d_min = np.minimum(dn, dd)
        return dn, dd, d_min

    def global_uncertainty(self, features: np.ndarray) -> np.ndarray:
        """Compute image-level global uncertainty Ug in [0, 1]."""

        _, _, d_min = self.nearest_distances(features)
        return _sigmoid(self.alpha * d_min - self.beta)

    def calibrate(self, reference_features: np.ndarray, low: float = 10.0, high: float = 90.0) -> None:
        """Set alpha and beta from reference distances.

        The calibration maps the midpoint between the chosen percentiles to
        sigmoid 0.5 and spreads the interval across a practical transition band.
        """

        _, _, d_min = self.nearest_distances(reference_features)
        lo, hi = np.percentile(d_min, [low, high])
        span = max(float(hi - lo), EPS)
        midpoint = float((hi + lo) / 2.0)
        self.alpha = 4.0 / span
        self.beta = self.alpha * midpoint

    def save(self, path: str | Path) -> None:
        """Save prototypes to a compact NumPy archive."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            normal=self.normal,
            defect=self.defect,
            alpha=np.array([self.alpha], dtype=np.float64),
            beta=np.array([self.beta], dtype=np.float64),
            normalize=np.array([int(self.normalize)], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeLibrary":
        """Load prototypes saved by :meth:`save`."""

        data = np.load(Path(path), allow_pickle=False)
        return cls(
            normal=data["normal"],
            defect=data["defect"],
            alpha=float(data["alpha"][0]),
            beta=float(data["beta"][0]),
            normalize=bool(int(data["normalize"][0])) if "normalize" in data else False,
        )


def build_prototype_library(
    features: np.ndarray,
    labels: Iterable[object],
    n_normal: int = 50,
    n_defect: int = 50,
    normal_label: object = "normal",
    seed: int = 0,
    normalize: bool = False,
    calibrate: bool = True,
) -> PrototypeLibrary:
    """Build a PGUQ prototype library with small NumPy K-means.

    Any label that does not equal ``normal_label`` is treated as a defect label.
    This keeps the package lightweight while still supporting multi-class
    defect datasets.
    """

    feats = _as_2d_float(features, "features")
    labels_array = np.asarray(list(labels), dtype=object)
    if len(labels_array) != len(feats):
        raise ValueError("labels length must match number of features")

    normal_mask = np.array([_same_label(v, normal_label) for v in labels_array], dtype=bool)
    if not np.any(normal_mask):
        raise ValueError(f"no samples matched normal_label={normal_label!r}")
    if np.all(normal_mask):
        raise ValueError("at least one defect sample is required")

    work = _l2_normalize(feats) if normalize else feats
    normal = _kmeans(work[normal_mask], n_normal, seed=seed)
    defect = _kmeans(work[~normal_mask], n_defect, seed=seed + 17)
    library = PrototypeLibrary(normal=normal, defect=defect, normalize=normalize)

    if calibrate:
        library.calibrate(feats)
    return library


def _same_label(value: object, normal_label: object) -> bool:
    return str(value).strip().lower() == str(normal_label).strip().lower()


def _as_2d_float(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _as_features(features: np.ndarray, feature_dim: int, normalize: bool) -> np.ndarray:
    feats = _as_2d_float(features, "features")
    if feats.shape[1] != feature_dim:
        raise ValueError(f"expected feature dimension {feature_dim}, got {feats.shape[1]}")
    return _l2_normalize(feats) if normalize else feats


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, EPS)


def _nearest_l2(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - prototypes[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return np.min(distances, axis=1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _kmeans(features: np.ndarray, k: int, seed: int, max_iter: int = 100) -> np.ndarray:
    """Small deterministic Lloyd K-means used to avoid a scikit-learn dependency."""

    feats = _as_2d_float(features, "features")
    k = max(1, min(int(k), len(feats)))
    rng = np.random.default_rng(seed)

    if len(feats) == k:
        return feats.copy()

    centers = feats[rng.choice(len(feats), size=k, replace=False)].copy()
    for _ in range(max_iter):
        diff = feats[:, None, :] - centers[None, :, :]
        labels = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        next_centers = centers.copy()
        for idx in range(k):
            members = feats[labels == idx]
            if len(members):
                next_centers[idx] = members.mean(axis=0)
        if np.allclose(next_centers, centers, rtol=1e-6, atol=1e-8):
            break
        centers = next_centers
    return centers
