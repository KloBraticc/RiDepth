# Model format

`export.py` writes a standard ONNX file, opset 17. With `--fp16` every weight and tensor is half precision, which halves memory traffic on the GPU.

## Input

| Name | Shape | Type |
| --- | --- | --- |
| `pixel_values` | `[batch_size, 3, height, width]` | float16 with `--fp16`, otherwise float32 |

RGB in the 0 to 1 range, normalised with the ImageNet mean `0.485, 0.456, 0.406` and standard deviation `0.229, 0.224, 0.225`. Any width and height works. RiShade uses sizes such as 266 by 154 and 518 by 294.

## Outputs

| Name | Shape | Meaning |
| --- | --- | --- |
| `predicted_depth` | `[batch_size, height, width]` | Relative closeness, higher is nearer, sky is 0, never negative |
| `normals` | `[batch_size, 3, height, width]` | Unit surface direction in camera space, x right, y up, z backward as in Roblox |

`predicted_depth` comes first, so tools that read only the first output get depth.

## Using it outside RiShade

Any ONNX Runtime build can run it. Name the dynamic axes `batch_size`, `height` and `width` when you fix them with free dimension overrides. Because depth is relative, scale it per frame, for example by dividing by a high percentile, before turning it into distance.
