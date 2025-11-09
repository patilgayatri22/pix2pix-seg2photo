import os, random
import numpy as np
import torch
from torchvision.utils import save_image as tv_save_image


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def denorm(x: torch.Tensor) -> torch.Tensor:
    # from [-1,1] to [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_image(tensor: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tv_save_image(denorm(tensor), path)
