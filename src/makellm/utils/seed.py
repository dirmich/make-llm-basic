"""재현성을 위한 시드 유틸."""

from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Python, NumPy, PyTorch 시드를 한 번에 설정.

    PyTorch CUDA 시드도 설정하지만 CUDA가 없으면 무시함.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
