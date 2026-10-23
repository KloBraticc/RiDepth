import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ridepth.data import GameplayDepth, RobloxDepth, find_gameplay, find_samples
from ridepth.losses import align_scale, depth_losses, normalize_target
from ridepth.model import VARIANTS, RiDepthNet

WEIGHTS = {"depth": 1.0, "gradient": 0.5, "normal": 0.5, "teacher_depth": 0.5, "teacher_normal": 0.25}


def teacher_losses(disparity, normals, teacher_disparity, teacher_normals):
    everything = torch.ones_like(teacher_disparity, dtype=torch.bool)
    target = normalize_target(teacher_disparity, everything)
    return {
        "teacher_depth": (align_scale(disparity, target) - target).abs().mean(),
        "teacher_normal": (1 - (normals * teacher_normals).sum(1)).mean(),
    }


def run_epoch(model, loader, device, optimizer=None, scaler=None, teacher=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = 0
    for image, disparity, normal, solid, exact in loader:
        image = image.to(device, non_blocking=True)
        disparity = disparity.to(device, non_blocking=True)
        normal = normal.to(device, non_blocking=True)
        solid = solid.to(device, non_blocking=True)
        exact = exact.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=device.type == "cuda"):
            predicted, normals = model(image)
        losses = depth_losses(predicted.float(), normals.float(), disparity, normal, solid, exact)
        if teacher is not None:
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                teacher_disparity, teacher_normals = teacher(image)
            losses.update(teacher_losses(predicted.float(), normals.float(), teacher_disparity.float(), teacher_normals.float()))
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
    parser.add_argument("--gameplay", type=Path, default=Path("gameplay"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    parser.add_argument("--teacher", type=Path)
    arguments = parser.parse_args()

    samples = find_samples(arguments.data)
    frames = find_gameplay(arguments.gameplay)
    if len(samples) + len(frames) < 20:
        raise SystemExit(f"found only {len(samples) + len(frames)} samples, capture more first")
    random.Random(7).shuffle(samples)
    random.Random(7).shuffle(frames)
    sample_split = min(len(samples), max(1, len(samples) // 10))
    frame_split = min(len(frames), max(1, len(frames) // 10))
    train_set = ConcatDataset([RobloxDepth(samples[sample_split:], augment=True), GameplayDepth(frames[frame_split:], augment=True)])
    validation_set = ConcatDataset([RobloxDepth(samples[:sample_split], augment=False), GameplayDepth(frames[:frame_split], augment=False)])
    print(f"training on {len(samples) - sample_split} Studio samples and {len(frames) - frame_split} gameplay frames, "
          f"validating on {sample_split} and {frame_split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VARIANTS[arguments.variant](pretrained=arguments.resume is None).to(device)
    if arguments.resume:
        model.load_state_dict(torch.load(arguments.resume, map_location=device, weights_only=True))
    teacher = None
    if arguments.teacher:
        teacher = RiDepthNet(pretrained=False).to(device)
        teacher.load_state_dict(torch.load(arguments.teacher, map_location=device, weights_only=True))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    train_loader = DataLoader(train_set, arguments.batch, shuffle=True, num_workers=arguments.workers, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_set, arguments.batch, num_workers=arguments.workers, pin_memory=True)
    arguments.runs.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        trained = run_epoch(model, train_loader, device, optimizer, scaler, teacher)
        validated = run_epoch(model, validation_loader, device)
        schedule.step()
        score = sum(WEIGHTS[name] * value for name, value in validated.items() if not name.startswith("teacher"))
        print(f"epoch {epoch}: train {trained} validation {validated}")
        torch.save(model.state_dict(), arguments.runs / "last.pt")
        if score < best:
            best = score
            torch.save(model.state_dict(), arguments.runs / "best.pt")
            print(f"new best validation score {best:.4f}")


if __name__ == "__main__":
    main()
