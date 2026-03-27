import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Any

def fix_all_seeds(seed: int = 42):
    """
    Fixes all relevant random seeds to ensure reproducible results.

    This sets seeds for Python's built-in `random` module, NumPy, and PyTorch,
    and enforces deterministic behavior in PyTorch operations.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass
class FoldResult:
    train_loss: list[float]
    val_loss: list[float]
    metric_history: list[float]
    best_metric: float
    best_model_state: dict
    model: Optional[Any] = None


@dataclass
class TaskResult:
    task_name: str
    q_statistic: float
    mean_score: float
    std_dev: float
    model: Optional[Any] = None