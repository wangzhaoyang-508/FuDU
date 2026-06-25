"""Fuzzy dual-dimensional uncertainty sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

LABELS = ("VL", "L", "H", "VH")
ACTIONS = ("DNS", "LS", "HS", "MS")
ACTION_PRIORITY = {"DNS": 0, "LS": 1, "HS": 2, "MS": 3}

DEFAULT_GLOBAL_SETS = {
    "VL": (0.0, 0.0, 0.10, 0.20),
    "L": (0.10, 0.20, 0.30, 0.40),
    "H": (0.30, 0.40, 0.60, 0.80),
    "VH": (0.60, 0.80, 1.00, 1.00),
}

DEFAULT_DEFECT_SETS = {
    "VL": (0.0, 0.0, 0.10, 0.20),
    "L": (0.10, 0.20, 0.50, 0.60),
    "H": (0.50, 0.60, 0.80, 0.90),
    "VH": (0.80, 0.90, 1.00, 1.00),
}

ELES_GLOBAL_SETS = {
    "VL": (0.0, 0.0, 0.10, 0.20),
    "L": (0.10, 0.20, 0.30, 0.40),
    "H": (0.30, 0.40, 0.55, 0.65),
    "VH": (0.55, 0.65, 1.00, 1.00),
}

ELES_DEFECT_SETS = {
    "VL": (0.0, 0.0, 0.08, 0.15),
    "L": (0.08, 0.15, 0.35, 0.45),
    "H": (0.35, 0.45, 0.70, 0.80),
    "VH": (0.70, 0.80, 1.00, 1.00),
}

DEFAULT_RULES = {
    ("VL", "VL"): "DNS",
    ("VL", "L"): "LS",
    ("VL", "H"): "HS",
    ("VL", "VH"): "MS",
    ("L", "VL"): "LS",
    ("L", "L"): "LS",
    ("L", "H"): "HS",
    ("L", "VH"): "MS",
    ("H", "VL"): "HS",
    ("H", "L"): "HS",
    ("H", "H"): "MS",
    ("H", "VH"): "MS",
    ("VH", "VL"): "MS",
    ("VH", "L"): "MS",
    ("VH", "H"): "MS",
    ("VH", "VH"): "MS",
}

DEFAULT_ACTION_PROBS = {
    "DNS": 0.0,
    "LS": 0.1,
    "HS": 0.5,
    "MS": 1.0,
}

PRESETS = {
    "nuclear_fuel_rod": (DEFAULT_GLOBAL_SETS, DEFAULT_DEFECT_SETS),
    "eles": (ELES_GLOBAL_SETS, ELES_DEFECT_SETS),
}


@dataclass(frozen=True)
class FuzzyDecision:
    global_uncertainty: float
    defect_uncertainty: float
    action: str
    probability: float
    global_membership: dict[str, float]
    defect_membership: dict[str, float]
    output_strengths: dict[str, float]


class FuzzySampler:
    """Mamdani-style fuzzy inference for FuDU sampling actions."""

    def __init__(
        self,
        input_sets: Mapping[str, tuple[float, float, float, float]] | None = None,
        global_sets: Mapping[str, tuple[float, float, float, float]] | None = None,
        defect_sets: Mapping[str, tuple[float, float, float, float]] | None = None,
        rules: Mapping[tuple[str, str], str] | None = None,
        action_probs: Mapping[str, float] | None = None,
    ) -> None:
        if input_sets is not None and (global_sets is not None or defect_sets is not None):
            raise ValueError("use either input_sets or global_sets/defect_sets, not both")
        self.global_sets = dict(global_sets or input_sets or DEFAULT_GLOBAL_SETS)
        self.defect_sets = dict(defect_sets or input_sets or DEFAULT_DEFECT_SETS)
        self.rules = dict(rules or DEFAULT_RULES)
        self.action_probs = dict(action_probs or DEFAULT_ACTION_PROBS)
        self._validate()

    def evaluate(self, global_uncertainty: float, defect_uncertainty: float) -> FuzzyDecision:
        """Evaluate sampling action and probability for one image."""

        ug = float(np.clip(global_uncertainty, 0.0, 1.0))
        ud = float(np.clip(defect_uncertainty, 0.0, 1.0))
        g_membership = self.membership(ug, kind="global")
        d_membership = self.membership(ud, kind="defect")
        output = {action: 0.0 for action in ACTIONS}

        for g_label, g_strength in g_membership.items():
            if g_strength <= 0:
                continue
            for d_label, d_strength in d_membership.items():
                if d_strength <= 0:
                    continue
                activation = min(g_strength, d_strength)
                action = self.rules[(g_label, d_label)]
                output[action] = max(output[action], activation)

        action = max(ACTIONS, key=lambda name: (output[name], ACTION_PRIORITY[name]))
        return FuzzyDecision(
            global_uncertainty=ug,
            defect_uncertainty=ud,
            action=action,
            probability=float(self.action_probs[action]),
            global_membership=g_membership,
            defect_membership=d_membership,
            output_strengths=output,
        )

    def membership(self, value: float, kind: str = "global") -> dict[str, float]:
        """Return trapezoidal membership for all linguistic labels."""

        x = float(np.clip(value, 0.0, 1.0))
        if kind == "global":
            sets = self.global_sets
        elif kind == "defect":
            sets = self.defect_sets
        else:
            raise ValueError("kind must be 'global' or 'defect'")
        return {name: trapezoid(x, *params) for name, params in sets.items()}

    def sample(self, global_uncertainty: float, defect_uncertainty: float, rng: np.random.Generator | None = None) -> tuple[bool, FuzzyDecision]:
        """Sample one image according to the inferred FuDU probability."""

        rng = rng or np.random.default_rng()
        decision = self.evaluate(global_uncertainty, defect_uncertainty)
        return bool(rng.random() < decision.probability), decision

    @classmethod
    def from_preset(cls, name: str) -> "FuzzySampler":
        """Create a sampler from a documented membership-parameter preset."""

        if name not in PRESETS:
            raise ValueError(f"unknown fuzzy preset {name!r}; expected one of {sorted(PRESETS)}")
        global_sets, defect_sets = PRESETS[name]
        return cls(global_sets=global_sets, defect_sets=defect_sets)

    def _validate(self) -> None:
        for name, sets in [("global_sets", self.global_sets), ("defect_sets", self.defect_sets)]:
            missing_sets = set(LABELS) - set(sets)
            if missing_sets:
                raise ValueError(f"missing {name}: {sorted(missing_sets)}")
        for key, action in self.rules.items():
            if key[0] not in LABELS or key[1] not in LABELS:
                raise ValueError(f"invalid rule key: {key}")
            if action not in ACTIONS:
                raise ValueError(f"invalid rule action: {action}")
        for action in ACTIONS:
            if action not in self.action_probs:
                raise ValueError(f"missing action probability: {action}")


def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership with support for left/right shoulder sets."""

    x = float(x)
    if not (a <= b <= c <= d):
        raise ValueError("expected a <= b <= c <= d")
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0

    if x < b:
        if b == a:
            return 1.0
        return float(np.clip((x - a) / (b - a), 0.0, 1.0))

    if x > c:
        if d == c:
            return 1.0
        return float(np.clip((d - x) / (d - c), 0.0, 1.0))

    return 0.0
