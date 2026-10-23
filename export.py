import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxconverter_common import float16

from ridepth.model import RiDepthNet

CHECK_SIZES = ((154, 266), (294, 518))


class Exported(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        disparity, normals = self.model(pixel_values)
        return disparity, normals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("exports/ridepth.onnx"))
    parser.add_argument("--fp16", action="store_true")
    arguments = parser.parse_args()

    model = RiDepthNet(pretrained=arguments.weights is None)
    if arguments.weights:
        model.load_state_dict(torch.load(arguments.weights, map_location="cpu"))
    model.eval()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, 3, *CHECK_SIZES[0])
    torch.onnx.export(
        Exported(model),
        (sample,),
        arguments.output,
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
    if arguments.fp16:
        converted = float16.convert_float_to_float16(onnx.load(arguments.output), keep_io_types=False)
        onnx.save(converted, arguments.output)

    session = onnxruntime.InferenceSession(str(arguments.output), providers=["CPUExecutionProvider"])
    dtype = np.float16 if arguments.fp16 else np.float32
    for height, width in CHECK_SIZES:
        outputs = session.run(None, {"pixel_values": np.random.randn(1, 3, height, width).astype(dtype)})
        if outputs[0].shape != (1, height, width) or outputs[1].shape != (1, 3, height, width):
            raise SystemExit(f"unexpected output shapes {[o.shape for o in outputs]} for input {height}x{width}")
        print(f"checked {height}x{width}: depth {outputs[0].shape}, normals {outputs[1].shape}")
    print(f"exported {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
