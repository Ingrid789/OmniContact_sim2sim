from pathlib import Path

import numpy as np

from common.path_config import PROJECT_ROOT


def resolve_project_path(path: str, fallback: str = "") -> str:
    path_obj = Path(str(path).strip() or fallback).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(PROJECT_ROOT) / path_obj
    return str(path_obj)


def sample_xy_around(
    center_xy: np.ndarray,
    min_radius: float,
    max_radius: float,
    *,
    y_positive: bool = False,
    angle_range_deg: float | None = None,
) -> np.ndarray:
    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    radius = float(np.random.uniform(float(min_radius), float(max_radius)))
    if angle_range_deg is not None:
        half_angle = np.deg2rad(float(angle_range_deg)) * 0.5
        theta = float(np.random.uniform(np.pi * 0.5 - half_angle, np.pi * 0.5 + half_angle))
    else:
        theta = float(np.random.uniform(0.0, np.pi) if y_positive else np.random.uniform(-np.pi, np.pi))
    return center_xy + radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)


def pose7(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float32).reshape(3),
            np.asarray(quat, dtype=np.float32).reshape(4),
        ]
    ).astype(np.float32)
