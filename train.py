import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ridepth.data import RobloxDepth, find_samples
from ridepth.losses import depth_losses
from ridepth.model import RiDepthNet

WEIGHTS = {"depth": 1.0, "gradient": 0.5, "normal": 0.5}


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = 0
    for image, disparity, normal, solid in loader:
        image = image.to(device, non_blocking=True)
        disparity = disparity.to(device, non_blocking=True)
        normal = normal.to(device, non_blocking=True)
        solid = solid.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=device.type == "cuda"):
            predicted, normals = model(image)
        losses = depth_losses(predicted.float(), normals.float(), disparity, normal, solid)
        loss = sum(WEIGHTS[name] * value for name, value in losses.items())
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value.item()
        count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()

    samples = find_samples(arguments.data)
    if len(samples) < 20:
        raise SystemExit(f"found only {len(samples)} samples in {arguments.data}, capture more with the Studio plugin first")
    random.Random(7).shuffle(samples)
    split = max(1, len(samples) // 10)
    train_set = RobloxDepth(samples[split:], augment=True)
    validation_set = RobloxDepth(samples[:split], augment=False)
    print(f"training on {len(train_set)} samples, validating on {len(validation_set)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiDepthNet(pretrained=arguments.resume is None).to(device)
    if arguments.resume:
        model.load_state_dict(torch.load(arguments.resume, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    train_loader = DataLoader(train_set, arguments.batch, shuffle=True, num_workers=arguments.workers, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_set, arguments.batch, num_workers=arguments.workers, pin_memory=True)
    arguments.runs.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        trained = run_epoch(model, train_loader, device, optimizer, scaler)
        validated = run_epoch(model, validation_loader, device)
        schedule.step()
        score = sum(WEIGHTS[name] * value for name, value in validated.items())
        print(f"epoch {epoch}: train {trained} validation {validated}")
        torch.save(model.state_dict(), arguments.runs / "last.pt")
        if score < best:
            best = score
            torch.save(model.state_dict(), arguments.runs / "best.pt")
            print(f"new best validation score {best:.4f}")


if __name__ == "__main__":
    main()
