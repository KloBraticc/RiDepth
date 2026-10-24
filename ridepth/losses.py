import torch
import torch.nn.functional as F

GRADIENT_SCALES = 4


def normalize_target(disparity, solid):
    scales = []
    for target, mask in zip(disparity, solid):
        values = target[mask]
        scales.append(torch.quantile(values, 0.95) if values.numel() > 16 else target.new_tensor(1.0))
    return disparity / torch.stack(scales).clamp(min=1e-6).view(-1, 1, 1)


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def align_scale(prediction, target, mask):
    numerator = (prediction * target * mask).flatten(1).sum(1)
    denominator = (prediction * prediction * mask).flatten(1).sum(1).clamp(min=1e-6)
    return prediction * (numerator / denominator).view(-1, 1, 1)


def align_scale_shift(prediction, target, mask):
    count = mask.flatten(1).sum(1).clamp(min=1)
    sum_p = (prediction * mask).flatten(1).sum(1)
    sum_t = (target * mask).flatten(1).sum(1)
    sum_pp = (prediction * prediction * mask).flatten(1).sum(1)
    sum_pt = (prediction * target * mask).flatten(1).sum(1)
    determinant = (count * sum_pp - sum_p * sum_p).clamp(min=1e-6)
    scale = (count * sum_pt - sum_p * sum_t) / determinant
    shift = (sum_t - scale * sum_p) / count
    return prediction * scale.view(-1, 1, 1) + shift.view(-1, 1, 1)


def gradient_loss(prediction, target, mask):
    total = prediction.new_tensor(0.0)
    for _ in range(GRADIENT_SCALES):
        difference = prediction - target
        total = total + masked_mean((difference[:, :, 1:] - difference[:, :, :-1]).abs(), mask[:, :, 1:] * mask[:, :, :-1])
        total = total + masked_mean((difference[:, 1:, :] - difference[:, :-1, :]).abs(), mask[:, 1:, :] * mask[:, :-1, :])
        prediction = F.avg_pool2d(prediction[:, None], 2)[:, 0]
        target = F.avg_pool2d(target[:, None], 2)[:, 0]
        mask = (F.avg_pool2d(mask[:, None], 2)[:, 0] > 0.999).float()
    return total / GRADIENT_SCALES


def normal_loss(prediction, target, solid):
    cosine = (prediction * target).sum(1)
    return ((1 - cosine) * solid).sum() / solid.sum().clamp(min=1)


def depth_losses(disparity, normals, target_disparity, target_normals, solid, exact):
    exact = exact.view(-1, 1, 1)
    target = normalize_target(target_disparity, solid)
    mask = (solid | exact).float()
    aligned = torch.where(exact, align_scale(disparity, target, mask), align_scale_shift(disparity, target, mask))
    return {
        "depth": masked_mean((aligned - target).abs(), mask),
        "gradient": gradient_loss(aligned, target, mask),
        "normal": normal_loss(normals, target_normals, (solid & exact).float()),
    }
