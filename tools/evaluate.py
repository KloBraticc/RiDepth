import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ridepth.data import MEAN, STD, GameplayDepth, RobloxDepth, find_gameplay, find_samples, split_gameplay
from ridepth.losses import SKY_LEVEL, align_scale, align_scale_shift, depth_losses, normalize_target
from ridepth.model import VARIANTS


def load(spec: str, device):
    variant, _, weights = spec.partition(":")
    model = VARIANTS[variant](pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


def score(model, dataset, device, batch):
    totals = {}
    count = 0
    with torch.no_grad():
        for image, disparity, normal, solid, exact in DataLoader(dataset, batch):
            image, disparity, normal, solid, exact = (tensor.to(device) for tensor in (image, disparity, normal, solid, exact))
            predicted, normals = model(image)
            losses = depth_losses(predicted.float(), normals.float(), disparity, normal, solid, exact)
            target = normalize_target(disparity, solid)
            mask = (solid | exact.view(-1, 1, 1)).float()
            aligned = torch.where(exact.view(-1, 1, 1), align_scale(predicted, target, mask), align_scale_shift(predicted, target, mask))
            near = (mask > 0) & (target > 0.05)
            ratio = torch.maximum(aligned.clamp(min=1e-6) / target.clamp(min=1e-6), target.clamp(min=1e-6) / aligned.clamp(min=1e-6))
            losses["delta1"] = (ratio[near] < 1.25).float().mean()
            sky = target < SKY_LEVEL
            own = normalize_target(predicted, solid & ~sky)
            losses["sky"] = own[sky].mean() if sky.any() else own.new_tensor(0.0)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item() * len(image)
            count += len(image)
    return {name: value / max(count, 1) for name, value in totals.items()}


def colorize(disparity):
    values = disparity / max(np.percentile(disparity, 98), 1e-6)
    return Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def previews(models, frames, folder: Path, device, per_game):
    folder.mkdir(parents=True, exist_ok=True)
    games = {}
    for frame in frames:
        games.setdefault(frame.parent.parent.name, []).append(frame)
    dataset = GameplayDepth([], augment=False)
    for game, items in sorted(games.items()):
        for frame in random.Random(3).sample(items, min(per_game, len(items))):
            image, disparity, _, _, _ = dataset.load(frame)
            tiles = [Image.fromarray(((image * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)), colorize(disparity.numpy())]
            with torch.no_grad():
                for model in models:
                    tiles.append(colorize(model(image[None].to(device))[0][0].float().cpu().numpy()))
            width, height = tiles[0].size
            sheet = Image.new("RGB", (width * len(tiles), height))
            for index, tile in enumerate(tiles):
                sheet.paste(tile, (index * width, 0))
            sheet.save(folder / f"{game}_{frame.stem}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="variant:weights, for example tiny:runs/tiny/best.pt")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--gameplay", type=Path, default=Path("gameplay"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--previews", type=Path)
    parser.add_argument("--per-game", type=int, default=3)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = find_samples(arguments.data)
    random.Random(7).shuffle(samples)
    studio = samples[: min(len(samples), max(1, len(samples) // 10))]
    _, held = split_gameplay(find_gameplay(arguments.gameplay))
    sets = {"studio": RobloxDepth(studio, augment=False), "unseen games": GameplayDepth(held, augment=False)}
    print(f"{len(studio)} Studio samples, {len(held)} frames from {sorted({frame.parent.parent.name for frame in held})}")

    models = [load(spec, device) for spec in arguments.models]
    for spec, model in zip(arguments.models, models):
        for name, dataset in sets.items():
            if len(dataset):
                metrics = score(model, dataset, device, arguments.batch)
                shown = "  ".join(f"{key} {value:.4f}" for key, value in metrics.items() if not (name != "studio" and key == "normal"))
                print(f"{spec:40s} {name:13s} {shown}")
    if arguments.previews:
        previews(models, held, arguments.previews, device, arguments.per_game)
        print(f"previews in {arguments.previews.resolve()}")


if __name__ == "__main__":
    main()
