"""
Central config for leJEPA scripts.

Keep this file simple: plain dataclasses with defaults, so all scripts can
import and reuse the same settings without duplicating constants everywhere.
"""

from dataclasses import dataclass


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


COMMON = CommonConfig()
COLLECT = CollectConfig()
TRAIN = TrainConfig()
INFER = InferenceConfig()
VIEW = ViewConfig()


