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
    return sorted(path for path in root.rglob("*.png") if path.name != "static.png" and path.with_suffix(".teacher.npy").exists())


def split_gameplay(frames, share=0.15, seed=7):
    games = sorted({frame.parent.parent.name for frame in frames})
    random.Random(seed).shuffle(games)
    held = set(games[: max(1, round(len(games) * share))] if games else [])
    return [frame for frame in frames if frame.parent.parent.name not in held], [frame for frame in frames if frame.parent.parent.name in held]


def read_image(path: Path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_image(path: Path, size):
    return F.interpolate(read_image(path)[None], size=size, mode="bilinear", align_corners=False)[0]


def random_crop(height, width, smallest=0.6):
    scale = random.uniform(smallest, 1.0) ** 0.5
    crop_h, crop_w = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = random.randint(0, height - crop_h), random.randint(0, width - crop_w)
    return slice(top, top + crop_h), slice(left, left + crop_w)


def recolor(image):
    brightness = random.uniform(0.75, 1.25)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.7, 1.3)
    grey = image.mean(0, keepdim=True)
    image = (image - grey) * saturation + grey
    image = (image - image.mean()) * contrast + image.mean()
    return (image * brightness).clamp(0, 1)


class RobloxDepth(Dataset):
    def __init__(self, samples, size=(294, 518), augment=True):
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
    def __init__(self, frames, size=(294, 518), augment=True):
        self.frames = list(frames)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.load(self.frames[index])

    def load(self, frame: Path):
        image = read_image(frame)
        height, width = image.shape[-2:]
        disparity = torch.from_numpy(np.load(frame.with_suffix(".teacher.npy")).astype(np.float32)).clamp(min=0)
        disparity = F.interpolate(disparity[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        static = frame.parent / "static.png"
        if static.exists():
            solid = torch.from_numpy(np.asarray(Image.open(static).convert("L").resize((width, height), Image.Resampling.NEAREST)) < 128)
        else:
            solid = torch.ones(height, width, dtype=torch.bool)
        if self.augment:
            rows, columns = random_crop(height, width)
            image, disparity, solid = image[:, rows, columns], disparity[rows, columns], solid[rows, columns]
            if random.random() < 0.5:
                image = image.flip(-1)
                disparity = disparity.flip(-1)
                solid = solid.flip(-1)
            image = recolor(image)
        image = F.interpolate(image[None], size=self.size, mode="bilinear", align_corners=False)[0]
        disparity = F.interpolate(disparity[None, None], size=self.size, mode="bilinear", align_corners=False)[0, 0]
        solid = F.interpolate(solid[None, None].float(), size=self.size, mode="nearest")[0, 0] > 0.5
        normal = torch.zeros(3, *self.size)
        return (image - MEAN) / STD, disparity, normal, solid, torch.tensor(False)
