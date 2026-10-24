# Training

## Models

| Variant | Encoder | Parameters | Decoder | Heads |
| --- | --- | --- | --- | --- |
| `large` (RiDepth Large) | Depth Anything V2 Small, DINOv2 ViT S | about 24.8 M | Depth Anything's DPT neck and head | depth from the original head, normals from an extra small head on the finest fused features |
| `base` (RiDepth) | MobileNetV3 Large, ImageNet | about 3.7 M | five levels, two convolutions each | depth and normals at full resolution |
| `tiny` (RiDepth Tiny) | MobileNetV3 Small, ImageNet | about 1.0 M | five levels, one convolution each | one 4 channel head at half resolution, then upsampled |

All of them return depth and normals at the input size. Large needs a width and height that are multiples of 14, which every size RiShade uses already is.

## Targets and losses

- **Depth** is trained as disparity, one over depth, with sky at 0. Each target is divided by its 95th percentile and the prediction is matched to it with a single best scale, so the model learns shape and order rather than absolute studs. Sky stays at 0 because no offset is allowed.
- **Gameplay frames** carry relative disparity from the teacher, which has an unknown offset, so the prediction is matched to them with a best scale and offset. Where the teacher puts sky at its far plane, a sky term pushes the model's output, divided by its own 95th percentile, towards 0, so open sky stays at 0 in real games as well as in Studio. Pixels under the interface mask are left out of every term.
- **Gradient loss** compares edges across four scales, which keeps outlines sharp.
- **Normal loss** is one minus the cosine between predicted and true surface direction, on solid pixels only.
- **Distillation** adds the same depth and normal losses against the predictions of a teacher model given with `--teacher` and `--teacher-variant`.

## Splits and checkpoints

A fixed 10 percent of the Studio samples and about 15 percent of the recorded games are held back for validation. Whole games are held back, never single frames, because frames from one game look alike and would make the score look better than it is. After every epoch the last weights go to `last.pt`, and whenever the validation score improves they also go to `best.pt`. The score only counts the depth, edge and normal terms. The distillation terms are left out so it always measures distance from the true depth, and the sky term is left out because far dark corners indoors look like sky to the teacher, which makes it too noisy to pick checkpoints by.

## Augmentation

Horizontal flips (the normal's sideways part is flipped with it) and random brightness, contrast and saturation. Gameplay frames are also cropped to between 60 and 100 percent of their area before resizing.

## Sampling

There are far more gameplay frames than Studio samples, so each batch is drawn with weights that keep Studio at a quarter of the training data by default. Change it with `--studio-share`.

## Evaluating

Compare checkpoints on the held out data, and optionally save previews of the frame, the teacher and every model side by side:

```bash
.venv\Scripts\python -m tools.evaluate base:runs\best.pt tiny:runs\tiny\best.pt --previews previews
```

It reports the aligned depth error, the edge error, the share of pixels within 25 percent of the target (`delta1`) and how close the model puts the sky to 0 (`sky`), separately for Studio and for unseen games.

## Tips

- With little data the training error drops far below the validation error. That means you need more places, not more epochs.
- Lower `--batch` if CUDA runs out of memory. 8 for `large`, 12 for `base` and 16 for `tiny` fit in 8 GB.
- Fine tune `large` with a low learning rate such as `5e-5`. Its backbone always trains ten times slower than the rest, so it keeps what Depth Anything already knows.
- Retrain a smaller variant after retraining its teacher, so it learns from the newest one.
