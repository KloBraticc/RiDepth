import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def find_samples(root: Path):
    return sorted(path.parent for path in root.rglob("meta.json") if (path.parent / "rgb.png").exists())


def find_gameplay(root: Path):
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.png") if path.with_suffix(".teacher.npy").exists())


def load_image(path: Path, size):
    image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]


def recolor(image):
    brightness = random.uniform(0.75, 1.25)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.7, 1.3)
    grey = image.mean(0, keepdim=True)
    image = (image - grey) * saturation + grey
    image = (image - image.mean()) * contrast + image.mean()
    return (image * brightness).clamp(0, 1)


class RobloxDepth(Dataset):
    def __init__(self, samples, size=(256, 448), augment=True):
        self.samples = list(samples)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        folder = self.samples[index]
        image = load_image(folder / "rgb.png", self.size)
        depth = torch.from_numpy(np.load(folder / "depth.npy").astype(np.float32))
        normal = torch.from_numpy(np.load(folder / "normal.npy").astype(np.float32))

        depth = F.interpolate(depth[None, None], size=self.size, mode="nearest")[0, 0]
        normal = F.interpolate(normal[None], size=self.size, mode="nearest")[0]

        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
                depth = depth.flip(-1)
                normal = normal.flip(-1)
                normal[0] = -normal[0]
            image = recolor(image)

        sky = depth <= 0
        disparity = torch.where(sky, torch.zeros_like(depth), 1.0 / depth.clamp(min=0.05))
        return (image - MEAN) / STD, disparity, normal, ~sky, torch.tensor(True)


class GameplayDepth(Dataset):
    def __init__(self, frames, size=(256, 448), augment=True):
        self.frames = list(frames)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        image = load_image(frame, self.size)
        disparity = torch.from_numpy(np.load(frame.with_suffix(".teacher.npy")).astype(np.float32)).clamp(min=0)
        disparity = F.interpolate(disparity[None, None], size=self.size, mode="bilinear", align_corners=False)[0, 0]
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
                disparity = disparity.flip(-1)
            image = recolor(image)
        normal = torch.zeros(3, *self.size)
        solid = torch.ones(self.size, dtype=torch.bool)
        return (image - MEAN) / STD, disparity, normal, solid, torch.tensor(False)
