# Training

## Models

| Variant | Encoder | Parameters | Decoder | Heads |
| --- | --- | --- | --- | --- |
| `base` (RiDepth) | MobileNetV3 Large, ImageNet | about 3.7 M | five levels, two convolutions each | depth and normals at full resolution |
| `tiny` (RiDepth Tiny) | MobileNetV3 Small, ImageNet | about 1.0 M | five levels, one convolution each | one 4 channel head at half resolution, then upsampled |

Both take any width and height and return depth and normals at the input size.

## Targets and losses

- **Depth** is trained as disparity, one over depth, with sky at 0. Each target is divided by its 95th percentile and the prediction is matched to it with a single best scale, so the model learns shape and order rather than absolute studs. Sky stays at 0 because no offset is allowed.
- **Gradient loss** compares edges across four scales, which keeps outlines sharp.
- **Normal loss** is one minus the cosine between predicted and true surface direction, on solid pixels only.
- **Distillation** (Tiny only) adds the same depth and normal losses against the teacher's predictions.

## Splits and checkpoints

A fixed 10 percent of samples is held back for validation. After every epoch the last weights go to `last.pt`, and whenever the validation score improves they also go to `best.pt`. The score ignores the distillation terms, so it always measures distance from the true depth.

## Augmentation

Horizontal flips (the normal's sideways part is flipped with it) and random brightness, contrast and saturation.

## Tips

- With little data the training error drops far below the validation error. That means you need more places, not more epochs.
- Lower `--batch` if CUDA runs out of memory. 12 for `base` and 16 for `tiny` fit in 8 GB.
- Retrain `tiny` after retraining `base`, so it learns from the newest teacher.
