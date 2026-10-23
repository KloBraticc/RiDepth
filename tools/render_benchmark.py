import argparse
import datetime
import hashlib
import json
import tempfile
import urllib.request
from html import escape
from pathlib import Path

import onnxruntime

from export import export_model
from tools.benchmark import SIZES, measure

REFERENCE = {
    "fp16": (
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model_fp16.onnx",
        "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04",
    ),
    "fp32": (
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx",
        "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
    ),
}
BENCH_SIZES = (266, 518)
THEMES = {
    "dark": {
        "background": "#3a3d42", "panel": "#45494f", "border": "#5c6168", "grid": "#4d5157", "text": "#e8e8e8",
        "muted": "#b4b8bd", "reference": "#9a9fa6", "base": "#c9a227", "tiny": "#5fb85f",
    },
    "light": {
        "background": "#d9dbde", "panel": "#cdd0d4", "border": "#a3a8ae", "grid": "#c3c6ca", "text": "#1e1e1e",
        "muted": "#4a4f55", "reference": "#7d8288", "base": "#b08a10", "tiny": "#2e8b2e",
    },
}
FONT = "Verdana, 'DejaVu Sans', Arial, sans-serif"


def fetch_reference(precision: str, cache: Path) -> Path:
    url, digest = REFERENCE[precision]
    target = cache / f"depth-anything-v2-small-{precision}.onnx"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit("the reference model download did not match its pinned hash")
        target.write_bytes(data)
    return target


def run(provider: str, precision: str, runs: int, warmup: int, cache: Path):
    fp16 = precision == "fp16"
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        models = {
            "Depth Anything V2 Small": fetch_reference(precision, cache),
            "RiDepth": export_model("base", None, work / "ridepth.onnx", fp16, pretrained=False),
            "RiDepth Tiny": export_model("tiny", None, work / "ridepth-tiny.onnx", fp16, pretrained=False),
        }
        results = []
        for name, path in models.items():
            timings = {}
            for size in BENCH_SIZES:
                height, width = SIZES[size]
                median, _ = measure(path, height, width, runs, warmup, provider)
                timings[str(size)] = round(median, 3)
            results.append({"name": name, "megabytes": round(path.stat().st_size / 1048576, 1), "milliseconds": timings})
    return results


def step(x, y, width, height, title, detail, theme):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{theme["panel"]}" stroke="{theme["border"]}" stroke-width="1"/>'
        f'<text x="{x + 10}" y="{y + 20}" font-size="12" font-weight="bold" fill="{theme["text"]}">{escape(title)}</text>'
        f'<text x="{x + 10}" y="{y + 37}" font-size="10.5" fill="{theme["muted"]}">{escape(detail)}</text>'
    )


def render(results, meta, theme_name):
    theme = THEMES[theme_name]
    width = 960
    tiny = next(item for item in results if item["name"] == "RiDepth Tiny")
    reference = next(item for item in results if item["name"] == "Depth Anything V2 Small")
    parts = [f'<rect width="{width}" height="HEIGHT" fill="{theme["background"]}"/>']
    parts.append(f'<text x="24" y="32" font-size="15" font-weight="bold" fill="{theme["text"]}">RiDepth pipeline</text>')
    steps = [
        ("Roblox Studio", "rays give the true depth"),
        ("capture_server.py", "saves picture + depth"),
        ("train.py", "PyTorch, CUDA"),
        ("export.py", f"Tiny ONNX, {tiny['megabytes']} MB"),
        ("RiShade", "DirectML, every frame"),
    ]
    box_width, gap, top = 166, 22, 46
    for index, (title, detail) in enumerate(steps):
        x = 24 + index * (box_width + gap)
        parts.append(step(x, top, box_width, 48, title, detail, theme))
        if index:
            parts.append(f'<text x="{x - gap / 2}" y="{top + 29}" text-anchor="middle" font-size="13" fill="{theme["muted"]}">&gt;</text>')

    chart_left, chart_right = 250, width - 140
    longest = max(max(item["milliseconds"].values()) for item in results)
    tick = next(value for value in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500) if longest / value <= 6)
    limit = tick * -(-longest // tick)
    scale = (chart_right - chart_left) / limit
    parts.append(f'<text x="24" y="138" font-size="15" font-weight="bold" fill="{theme["text"]}">Time per depth frame (ms, lower is better)</text>')
    y = 160
    groups_top = y
    rows = []
    for size in BENCH_SIZES:
        height, image_width = SIZES[size]
        rows.append(f'<text x="24" y="{y + 12}" font-size="12" font-weight="bold" fill="{theme["text"]}">{image_width} x {height} input</text>')
        y += 20
        for item in results:
            value = item["milliseconds"][str(size)]
            span = value * scale
            color = {"Depth Anything V2 Small": theme["reference"], "RiDepth": theme["base"], "RiDepth Tiny": theme["tiny"]}[item["name"]]
            rows.append(f'<text x="{chart_left - 8}" y="{y + 11}" text-anchor="end" font-size="11" fill="{theme["text"]}">{escape(item["name"])} ({item["megabytes"]} MB)</text>')
            rows.append(f'<rect x="{chart_left}" y="{y}" width="{max(span, 1):.1f}" height="14" fill="{color}"/>')
            label = f"{value:.2f}"
            if item is not reference:
                label += f" ({reference['milliseconds'][str(size)] / value:.1f}x)"
            rows.append(f'<text x="{chart_left + span + 6:.1f}" y="{y + 11}" font-size="11" fill="{theme["text"]}">{escape(label)}</text>')
            y += 20
        y += 12
    axis = y
    for index in range(int(limit // tick) + 1):
        x = chart_left + index * tick * scale
        parts.append(f'<line x1="{x:.1f}" y1="{groups_top}" x2="{x:.1f}" y2="{axis}" stroke="{theme["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis + 14}" text-anchor="middle" font-size="10" fill="{theme["muted"]}">{index * tick:g}</text>')
    parts.append(f'<line x1="{chart_left}" y1="{axis}" x2="{chart_right}" y2="{axis}" stroke="{theme["border"]}" stroke-width="1"/>')
    parts.extend(rows)
    footer = f"{meta['date']}, {meta['device']}, ONNX Runtime {meta['runtime']}, {meta['precision']}, median of {meta['runs']} runs. Updated by the benchmark workflow."
    parts.append(f'<text x="24" y="{axis + 38}" font-size="10.5" fill="{theme["muted"]}">{escape(footer)}</text>')
    total = axis + 52
    body = "".join(parts).replace("HEIGHT", str(total))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total}" viewBox="0 0 {width} {total}" role="img" '
        f'aria-label="RiDepth pipeline and speed benchmark" font-family="{FONT}">{body}</svg>'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="CPUExecutionProvider", choices=["CPUExecutionProvider", "DmlExecutionProvider"])
    parser.add_argument("--precision", default="fp32", choices=sorted(REFERENCE))
    parser.add_argument("--device", default="a GitHub Actions Windows runner CPU")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "ridepth-benchmark")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    arguments = parser.parse_args()

    results = run(arguments.provider, arguments.precision, arguments.runs, arguments.warmup, arguments.cache)
    meta = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "device": arguments.device,
        "runtime": onnxruntime.__version__,
        "precision": "half precision" if arguments.precision == "fp16" else "full precision",
        "runs": arguments.runs,
    }
    arguments.assets.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (arguments.assets / f"benchmark-{theme}.svg").write_text(render(results, meta, theme), encoding="utf-8")
    (arguments.assets / "benchmark.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n", encoding="utf-8")
    for item in results:
        print(item["name"], item["megabytes"], "MB", item["milliseconds"])


if __name__ == "__main__":
    main()
