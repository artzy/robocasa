"""Interpolation helpers for live preview timelines."""

from __future__ import annotations

import numpy as np


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def lerp_vec(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a + smoothstep(alpha) * (b - a)


def lerp_scalar(a: float, b: float, alpha: float) -> float:
    return float(a + smoothstep(alpha) * (b - a))


def lerp_yaw(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    az = float(a[2])
    bz = float(b[2])
    diff = (bz - az + np.pi) % (2.0 * np.pi) - np.pi
    out = np.array(a, dtype=float).copy()
    out[2] = az + smoothstep(alpha) * diff
    return out
