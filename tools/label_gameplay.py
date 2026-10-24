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
STATIC_SIZE = (160, 90)
STATIC_TOLERANCE = 6 / 255
STATIC_SHARE = 0.6
STATIC_EDGE = 0.06
STATIC_REACH = 4
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


def mark_static(session: Path):
    frames = sorted(session.glob("*.png"))
    frames = [frame for frame in frames if frame.name != "static.png"]
    if len(frames) < 8:
        return None
    stack = np.stack([np.asarray(Image.open(frame).convert("RGB").resize(STATIC_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0 for frame in frames])
    median = np.median(stack, axis=0)
    matches = (np.abs(stack - median).max(axis=-1) < STATIC_TOLERANCE).mean(axis=0)
    grey = median.mean(axis=-1)
    edges = np.zeros_like(grey, dtype=bool)
    edges[:, 1:] |= np.abs(np.diff(grey, axis=1)) > STATIC_EDGE
    edges[1:, :] |= np.abs(np.diff(grey, axis=0)) > STATIC_EDGE
    static = (matches > STATIC_SHARE) & grow(edges, STATIC_REACH)
    static = grow(static, 1)
    Image.fromarray((static * 255).astype(np.uint8)).save(session / "static.png")
    return static.mean()


def grow(mask, reach):
    padded = np.pad(mask, reach)
    grown = np.zeros_like(mask)
    for dy in range(2 * reach + 1):
        for dx in range(2 * reach + 1):
            grown |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return grown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("gameplay"))
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "ridepth-teacher")
    parser.add_argument("--relabel", action="store_true")
    arguments = parser.parse_args()
    sessions = sorted({path.parent for path in arguments.frames.rglob("*.png")})
    for session in sessions:
        share = mark_static(session)
        if share is not None:
            print(f"{session.relative_to(arguments.frames)}: {share * 100:.1f} percent static interface")
    frames = sorted(path for path in arguments.frames.rglob("*.png") if path.name != "static.png" and (arguments.relabel or not path.with_suffix(".teacher.npy").exists()))
    if not frames:
        print("every frame is already labelled")
        return
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
        np.save(frame.with_suffix(".teacher.npy"), disparity.astype(np.float16))
        if index % 100 == 0 or index == len(frames):
            print(f"labelled {index} of {len(frames)}")


if __name__ == "__main__":
    main()
