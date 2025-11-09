from pathlib import Path
from typing import Tuple
import random, os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RandomJitter:
    """
    Resize a bit larger, random-crop back to size, optional hflip.
    """
    def __init__(self, size: int = 256, resize_to: int = 286):
        self.size = size
        self.resize_to = resize_to

    def __call__(self, img: Image.Image) -> Image.Image:
        img = transforms.Resize((self.resize_to, self.resize_to))(img)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.size, self.size))
        img = transforms.functional.crop(img, i, j, h, w)
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        return img


class SideBySideDataset(Dataset):
    """
    Expects images shaped as [photo | segmentation] on a single canvas.
    For seg->photo, we use RIGHT half as condition, LEFT half as target.
    """
    def __init__(self, root_dir: str, augment: bool = True, img_size: int = 256):
        self.files = sorted(
            [str(Path(root_dir) / f) for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
        )
        self.augment = augment
        self.rj = RandomJitter(size=img_size, resize_to=286)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        w2 = w // 2

        # left=photo, right=seg
        photo = img.crop((0, 0, w2, h))
        seg   = img.crop((w2, 0, w, h))

        cond, target = seg, photo  # seg -> photo

        if self.augment:
            # Use a shared seed so cond/target get identical spatial transforms
            seed = torch.randint(0, 2**32, (1,)).item()
            random.seed(seed); cond   = self.rj(cond)
            random.seed(seed); target = self.rj(target)

        cond   = self.to_tensor(cond)
        target = self.to_tensor(target)
        name = Path(p).name
        return cond, target, name
