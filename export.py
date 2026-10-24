import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from ridepth.model import VARIANTS

CHECK_SIZES = ((154, 266), (294, 518))


class Exported(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        disparity, normals = self.model(pixel_values)
        return disparity, normals


def export_model(variant: str, weights: Path | None, output: Path, fp16: bool, pretrained: bool = True) -> Path:
    model = VARIANTS[variant](pretrained=pretrained and weights is None)
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if fp16 and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    sample = torch.randn(1, 3, *CHECK_SIZES[0], device=device)
    if fp16:
        model, sample = model.half(), sample.half()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Exported(model),
        (sample,),
        output,
        input_names=["pixel_values"],
        output_names=["predicted_depth", "normals"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "predicted_depth": {0: "batch_size", 1: "height", 2: "width"},
            "normals": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    dtype = np.float16 if fp16 else np.float32
    for height, width in CHECK_SIZES:
        outputs = session.run(None, {"pixel_values": np.random.randn(1, 3, height, width).astype(dtype)})
        if outputs[0].shape != (1, height, width) or outputs[1].shape != (1, 3, height, width):
            raise SystemExit(f"unexpected output shapes {[o.shape for o in outputs]} for input {height}x{width}")
        print(f"checked {height}x{width}: depth {outputs[0].shape}, normals {outputs[1].shape}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    parser.add_argument("--output", type=Path, default=Path("exports/ridepth.onnx"))
    parser.add_argument("--fp16", action="store_true")
    arguments = parser.parse_args()
    exported = export_model(arguments.variant, arguments.weights, arguments.output, arguments.fp16)
    print(f"exported {exported.resolve()}")


if __name__ == "__main__":
    main()
