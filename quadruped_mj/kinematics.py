from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegGeometry:
    hip_offset_y: float
    upper_leg: float
    lower_leg: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def leg_ik(
    foot_position_body: np.ndarray,
    is_left: bool,
    geometry: LegGeometry,
) -> np.ndarray:
    """Solve a simple 3-DoF leg IK in the leg-local frame.

    The leg frame origin is the hip abduction joint.
    x: forward
    y: left
    z: up
    """
    x, y, z = foot_position_body.astype(float)

    side_sign = 1.0 if is_left else -1.0
    y_rel = y - side_sign * geometry.hip_offset_y

    hip_roll = math.atan2(y_rel, -z)

    sagittal_dist = math.sqrt(x * x + z * z)
    reach = clamp(
        sagittal_dist,
        1e-6,
        geometry.upper_leg + geometry.lower_leg - 1e-6,
    )

    cos_knee = clamp(
        (reach * reach - geometry.upper_leg**2 - geometry.lower_leg**2)
        / (2.0 * geometry.upper_leg * geometry.lower_leg),
        -1.0,
        1.0,
    )
    knee = -math.acos(cos_knee)

    hip_to_foot = math.atan2(-x, -z)
    knee_contrib = math.atan2(
        geometry.lower_leg * math.sin(knee),
        geometry.upper_leg + geometry.lower_leg * math.cos(knee),
    )
    hip_pitch = hip_to_foot - knee_contrib

    return np.array([hip_roll, hip_pitch, knee], dtype=float)
