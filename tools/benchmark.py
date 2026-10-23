import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime

SIZES = {266: (154, 266), 392: (224, 392), 518: (294, 518)}
TYPES = {"tensor(float16)": np.float16, "tensor(float)": np.float32}


DEVICES = {"DmlExecutionProvider": "dml", "CPUExecutionProvider": "cpu"}


def measure(path: Path, height: int, width: int, runs: int, warmup: int, provider: str = "DmlExecutionProvider"):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    for name, value in (("batch_size", 1), ("num_images", 1), ("height", height), ("width", width)):
        options.add_free_dimension_override_by_name(name, value)
    session = onnxruntime.InferenceSession(str(path), options, providers=[provider])
    device = DEVICES[provider]
    source = session.get_inputs()[0]
    shape = [1, 3, height, width] if len(source.shape) == 4 else [1, 1, 3, height, width]
    pixels = onnxruntime.OrtValue.ortvalue_from_numpy(np.random.randn(*shape).astype(TYPES[source.type]), device, 0)
    binding = session.io_binding()
    binding.bind_ortvalue_input(source.name, pixels)
    binding.bind_output(session.get_outputs()[0].name, device, 0)
    for _ in range(warmup):
        session.run_with_iobinding(binding)
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings), min(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--sizes", type=int, nargs="+", default=[266, 518])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--provider", choices=sorted(DEVICES), default="DmlExecutionProvider")
    arguments = parser.parse_args()
    for model in arguments.models:
        megabytes = sum(file.stat().st_size for file in model.parent.glob(model.name + "*")) / 1048576
        for size in arguments.sizes:
            height, width = SIZES[size]
            median, best = measure(model, height, width, arguments.runs, arguments.warmup, arguments.provider)
            print(f"{model.name:40s} {megabytes:7.1f} MB  {width}x{height}  median {median:7.2f} ms  best {best:7.2f} ms")


if __name__ == "__main__":
    main()
