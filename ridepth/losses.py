import torch
import torch.nn.functional as F

GRADIENT_SCALES = 4


def normalize_target(disparity, solid):
    scales = []
    for target, mask in zip(disparity, solid):
        values = target[mask]
        scales.append(torch.quantile(values, 0.95) if values.numel() > 16 else target.new_tensor(1.0))
    return disparity / torch.stack(scales).clamp(min=1e-6).view(-1, 1, 1)


def align_scale(prediction, target):
    numerator = (prediction * target).flatten(1).sum(1)
    denominator = (prediction * prediction).flatten(1).sum(1).clamp(min=1e-6)
    return prediction * (numerator / denominator).view(-1, 1, 1)


def align_scale_shift(prediction, target):
    count = prediction[0].numel()
    sum_p = prediction.flatten(1).sum(1)
    sum_t = target.flatten(1).sum(1)
    sum_pp = (prediction * prediction).flatten(1).sum(1)
    sum_pt = (prediction * target).flatten(1).sum(1)
    determinant = (count * sum_pp - sum_p * sum_p).clamp(min=1e-6)
    scale = (count * sum_pt - sum_p * sum_t) / determinant
    shift = (sum_t - scale * sum_p) / count
    return prediction * scale.view(-1, 1, 1) + shift.view(-1, 1, 1)


def gradient_loss(prediction, target):
    total = prediction.new_tensor(0.0)
    for _ in range(GRADIENT_SCALES):
        difference = prediction - target
        total = total + (difference[:, :, 1:] - difference[:, :, :-1]).abs().mean()
        total = total + (difference[:, 1:, :] - difference[:, :-1, :]).abs().mean()
        prediction = F.avg_pool2d(prediction[:, None], 2)[:, 0]
        target = F.avg_pool2d(target[:, None], 2)[:, 0]
    return total / GRADIENT_SCALES


def normal_loss(prediction, target, solid):
    cosine = (prediction * target).sum(1)
    return ((1 - cosine) * solid).sum() / solid.sum().clamp(min=1)


def depth_losses(disparity, normals, target_disparity, target_normals, solid, exact):
    target = normalize_target(target_disparity, solid)
    aligned = torch.where(exact.view(-1, 1, 1), align_scale(disparity, target), align_scale_shift(disparity, target))
    return {
        "depth": (aligned - target).abs().mean(),
        "gradient": gradient_loss(aligned, target),
        "normal": normal_loss(normals, target_normals, (solid & exact.view(-1, 1, 1)).float()),
    }
