"""
Central config for leJEPA scripts.

Keep this file simple: plain dataclasses with defaults, so all scripts can
import and reuse the same settings without duplicating constants everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import yaml


@dataclass
class CommonConfig:
    mujoco_gl: str = "egl"
    task_name: str = "push-v3"
    camera_name: str = "corner"
    img_size: tuple[int, int] = (64, 64)
    render_size: tuple[int, int] = (128, 128)
    data_dir: str = "./data_push_v3"
    model_path: str = "lejepa_robot.pth"


@dataclass
class CollectConfig:
    num_episodes: int = 5000
    max_steps: int = 100
    noise_scale: float = 0.3
    epsilon: float = 0.2
    stream_to_npy: bool = True
    preview: bool = False
    preview_wait_ms: int = 1
    preview_scale: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 5e-4
    epochs: int = 20
    lambda_coef: float = 0.05
    use_dataparallel: bool = True


@dataclass
class InferenceConfig:
    # MPPI
    horizon: int = 50
    num_samples: int = 2048
    temperature: float = 0.1

    # Reset/episode control (giống 01_collect_data.py)
    num_episodes: int = 100
    max_steps: int = 200

    # Visual
    preview: bool = False
    preview_wait_ms: int = 1
    preview_scale: int = 6

    # Multi-GPU
    use_dataparallel: bool = True


@dataclass
class ViewConfig:
    default_scale: int = 6
    default_stride: int = 1


def _apply_overrides(obj, overrides: dict | None):
    if not overrides:
        return obj
    for k, v in overrides.items():
        if not hasattr(obj, k):
            continue
        # normalize list -> tuple for sizes
        if k in ("img_size", "render_size") and isinstance(v, list):
            v = tuple(v)
        setattr(obj, k, v)
    return obj


def _load_yaml_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


_CFG = _load_yaml_config()

COMMON = _apply_overrides(CommonConfig(), _CFG.get("common"))
COLLECT = _apply_overrides(CollectConfig(), _CFG.get("collect"))
TRAIN = _apply_overrides(TrainConfig(), _CFG.get("train"))
INFER = _apply_overrides(InferenceConfig(), _CFG.get("infer"))
VIEW = _apply_overrides(ViewConfig(), _CFG.get("view"))


