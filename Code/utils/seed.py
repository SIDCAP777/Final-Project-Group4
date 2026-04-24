# ============================================================
# seed.py
# Sets random seeds across all libraries for reproducibility.
# Call set_seed(seed) once at the start of every script.
# ============================================================

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all GPUs if multi-GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Make PyTorch deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed] All random seeds set to {seed}")
