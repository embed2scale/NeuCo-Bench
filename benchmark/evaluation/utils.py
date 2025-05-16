import random, numpy as np, torch

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