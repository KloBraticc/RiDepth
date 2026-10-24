import torch
import torch.nn.functional as F

GRADIENT_SCALES = 4
SKY_LEVEL = 0.02
EDGE_JUMP = 0.08


def percentile_scale(disparity, solid):
    scales = []
    for target, mask in zip(disparity, solid):
        values = target[mask]
        scales.append(torch.quantile(values, 0.95) if values.numel() > 16 else target.new_tensor(1.0))
    return torch.stack(scales).clamp(min=1e-6).view(-1, 1, 1)


def normalize_target(disparity, solid):
    return disparity / percentile_scale(disparity, solid)


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


def edge_loss(prediction, target, mask):
    horizontal = mask[:, :, 1:] * mask[:, :, :-1]
    vertical = mask[:, 1:, :] * mask[:, :-1, :]
    target_x = target[:, :, 1:] - target[:, :, :-1]
    target_y = target[:, 1:, :] - target[:, :-1, :]
    scale = target.flatten(1).abs().mean(1).clamp(min=1e-6).view(-1, 1, 1)
    weight_x = horizontal * (target_x.abs() / scale > EDGE_JUMP).float()
    weight_y = vertical * (target_y.abs() / scale > EDGE_JUMP).float()
    error_x = ((prediction[:, :, 1:] - prediction[:, :, :-1]) - target_x).abs()
    error_y = ((prediction[:, 1:, :] - prediction[:, :-1, :]) - target_y).abs()
    return masked_mean(error_x, weight_x) + masked_mean(error_y, weight_y)


def planar_loss(prediction, target, mask):
    difference = prediction - target
    laplacian = difference[:, 1:-1, 1:-1] * 4 - difference[:, :-2, 1:-1] - difference[:, 2:, 1:-1] - difference[:, 1:-1, :-2] - difference[:, 1:-1, 2:]
    inner = mask[:, 1:-1, 1:-1] * mask[:, :-2, 1:-1] * mask[:, 2:, 1:-1] * mask[:, 1:-1, :-2] * mask[:, 1:-1, 2:]
    return masked_mean(laplacian.abs(), inner)


def normal_loss(prediction, target, solid):
    cosine = (prediction * target).sum(1)
    return ((1 - cosine) * solid).sum() / solid.sum().clamp(min=1)


def depth_losses(disparity, normals, target_disparity, target_normals, solid, exact):
    exact = exact.view(-1, 1, 1)
    target = normalize_target(target_disparity, solid)
    mask = (solid | exact).float()
    aligned = torch.where(exact, align_scale(disparity, target, mask), align_scale_shift(disparity, target, mask))
    sky = solid & ~exact & (target < SKY_LEVEL)
    return {
        "depth": masked_mean((aligned - target).abs(), mask),
        "gradient": gradient_loss(aligned, target, mask),
        "normal": normal_loss(normals, target_normals, (solid & exact).float()),
        "edge": edge_loss(aligned, target, mask),
        "planar": planar_loss(aligned, target, (solid & exact).float()),
        "sky": masked_mean(disparity / percentile_scale(disparity.detach(), solid & ~sky), sky.float()),
    }
