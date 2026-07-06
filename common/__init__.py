"""项目公共工具：路径、设备检测。

数据加载见 common.data，实验辅助（训练循环、指标、日志）见 common.experiment。
"""

from pathlib import Path

import torch

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    """检测可用的最佳硬件加速器（CUDA > MPS > CPU）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_tf32(device: torch.device) -> None:
    """在 Ampere 及以上 GPU（如 RTX 3080 Ti）上启用 TF32 与 cuDNN benchmark。"""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
