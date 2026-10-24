import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ridepth.data import GameplayDepth, RobloxDepth, find_gameplay, find_samples, split_gameplay
from ridepth.losses import align_scale, depth_losses, normalize_target
from ridepth.model import VARIANTS, RiDepthNet

WEIGHTS = {"depth": 1.0, "gradient": 0.5, "normal": 0.5, "sky": 0.5, "teacher_depth": 0.5, "teacher_normal": 0.25}
SCORED = ("depth", "gradient", "normal")


def teacher_losses(disparity, normals, teacher_disparity, teacher_normals):
    everything = torch.ones_like(teacher_disparity, dtype=torch.bool)
    target = normalize_target(teacher_disparity, everything)
    return {
        "teacher_depth": (align_scale(disparity, target, everything.float()) - target).abs().mean(),
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
    parser.add_argument("--studio-share", type=float, default=0.25)
    arguments = parser.parse_args()

    samples = find_samples(arguments.data)
    frames = find_gameplay(arguments.gameplay)
    if len(samples) + len(frames) < 20:
        raise SystemExit(f"found only {len(samples) + len(frames)} samples, capture more first")
    random.Random(7).shuffle(samples)
    sample_split = min(len(samples), max(1, len(samples) // 10))
    train_frames, held_frames = split_gameplay(frames)
    studio_train = RobloxDepth(samples[sample_split:], augment=True)
    gameplay_train = GameplayDepth(train_frames, augment=True)
    train_set = ConcatDataset([studio_train, gameplay_train])
    validation_set = ConcatDataset([RobloxDepth(samples[:sample_split], augment=False), GameplayDepth(held_frames, augment=False)])
    held_games = sorted({frame.parent.parent.name for frame in held_frames})
    print(f"training on {len(studio_train)} Studio samples and {len(gameplay_train)} gameplay frames, "
          f"validating on {sample_split} Studio samples and {len(held_frames)} frames from unseen games {held_games}")

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

    share = arguments.studio_share if len(studio_train) and len(gameplay_train) else 0.5
    weights = [share / max(len(studio_train), 1)] * len(studio_train) + [(1 - share) / max(len(gameplay_train), 1)] * len(gameplay_train)
    sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
    train_loader = DataLoader(train_set, arguments.batch, sampler=sampler, num_workers=arguments.workers, pin_memory=True,
                              drop_last=True, persistent_workers=arguments.workers > 0)
    validation_loader = DataLoader(validation_set, arguments.batch, num_workers=arguments.workers, pin_memory=True,
                                   persistent_workers=arguments.workers > 0)
    arguments.runs.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        trained = run_epoch(model, train_loader, device, optimizer, scaler, teacher)
        validated = run_epoch(model, validation_loader, device)
        schedule.step()
        score = sum(WEIGHTS[name] * value for name, value in validated.items() if name in SCORED)
        print(f"epoch {epoch}: train {trained} validation {validated}")
        torch.save(model.state_dict(), arguments.runs / "last.pt")
        if score < best:
            best = score
            torch.save(model.state_dict(), arguments.runs / "best.pt")
            print(f"new best validation score {best:.4f}")


if __name__ == "__main__":
    main()
