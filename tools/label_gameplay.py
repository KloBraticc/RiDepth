import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

TEACHER_URL = "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/1fa1591c7b080e98da9655827c3a33a3972a4a83/onnx/model_fp16.onnx"
TEACHER_SHA256 = "8eefc7afb877b10d413c8e1024b012073c6e4f62735d39555b1083fee3aabf0a"
TEACHER_SIZE = (518, 294)
LABEL_SIZE = (320, 180)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CHUNK = 1 << 20


def fetch_teacher(cache: Path) -> Path:
    target = cache / "depth-anything-v2-large-fp16.onnx"
    if target.exists() and sha256(target) == TEACHER_SHA256:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print("downloading the Depth Anything V2 Large teacher, about 640 MB")
    with urllib.request.urlopen(TEACHER_URL, timeout=120) as response, partial.open("wb") as file:
        while chunk := response.read(CHUNK):
            file.write(chunk)
    if sha256(partial) != TEACHER_SHA256:
        partial.unlink()
        raise SystemExit("the teacher download did not match its pinned hash")
    partial.replace(target)
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("gameplay"))
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "ridepth-teacher")
    arguments = parser.parse_args()
    frames = sorted(path for path in arguments.frames.rglob("*.png") if not path.with_suffix(".teacher.npy").exists())
    if not frames:
        raise SystemExit(f"no unlabelled frames in {arguments.frames}")
    options = onnxruntime.SessionOptions()
    width, height = TEACHER_SIZE
    for name, value in (("batch_size", 1), ("height", height), ("width", width)):
        options.add_free_dimension_override_by_name(name, value)
    session = onnxruntime.InferenceSession(str(fetch_teacher(arguments.cache)), options, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    source = session.get_inputs()[0]
    dtype = np.float16 if source.type == "tensor(float16)" else np.float32
    print(f"labelling {len(frames)} frames with {session.get_providers()[0]}")
    for index, frame in enumerate(frames, 1):
        image = Image.open(frame).convert("RGB").resize(TEACHER_SIZE, Image.Resampling.BICUBIC)
        pixels = (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
        disparity = session.run(None, {source.name: pixels[None].astype(dtype)})[0].reshape(height, width).astype(np.float32)
        label = Image.fromarray(disparity).resize(LABEL_SIZE, Image.Resampling.BILINEAR)
        np.save(frame.with_suffix(".teacher.npy"), np.asarray(label, dtype=np.float16))
        if index % 100 == 0 or index == len(frames):
            print(f"labelled {index} of {len(frames)}")


if __name__ == "__main__":
    main()
