<h1 align="center"><b>RiDepth</b></h1>

<p align="center">
  A depth model made for Roblox, trained on exact depth from Roblox Studio and on real gameplay.
</p>

<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/benchmark-dark.svg"><img src="assets/benchmark-light.svg" alt="RiDepth pipeline and speed benchmark" width="820"/></picture>
</p>

<p align="center">
  <a href="#capturing-training-data">Get started</a> |
  <a href="https://github.com/KloBraticc/Voidstrap">Voidstrap</a> |
  <a href="docs">Docs</a> |
  <a href="LICENSE">License</a>
</p>

<div align="center">

[![License][shield-license-flat]][link-license]
[![Stars][shield-repo-stars]][repo-stargazers]
[![Last commit][shield-last-commit]][repo-commits]

</div>

<h5 align="center">
  Leave a star if you like the project! ⭐️
</h5>

<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/windows-tile-dark.svg"><img src="assets/readme/windows-tile-light.svg" alt="Windows" width="32" height="32"/></picture>
</p>

---

> [!NOTE]
> RiDepth is the depth model behind RiShade in [Voidstrap](https://github.com/KloBraticc/Voidstrap). General depth models are trained on photos, so on Roblox they blur edges, put the sky at the wrong distance and leave ghosts in reflections. RiDepth Large starts from Depth Anything V2 Small and is fine tuned on Roblox only, with depth that Studio measures exactly and frames from real games.

The chart above is rebuilt by GitHub Actions from a fresh benchmark run, so the numbers are always real. It runs on a plain CPU runner, where everything is much slower than on a GPU. On an RTX 3060 Ti through DirectML, RiDepth Large takes about 2.4 ms per frame at 266 px, 4 ms at 392 px and 8 ms at 518 px.

## Features

- **Exact training data**: a Studio plugin casts a ray for every pixel, so each sample has true depth and true surface direction
- **Real gameplay**: frames recorded in many games are labelled by Depth Anything V2 Large, with the game's interface masked out
- **Two outputs**: depth for every effect, and surface normals that reflections can use directly instead of guessing them from depth
- **Sky at the far plane**: open sky is learned as infinitely far, so reflections never smear into it
- **RiDepth Large**: Depth Anything V2 Small (24.8 million parameters, Apache 2.0) fine tuned on Roblox, about 48 MB in half precision, with a full precision copy for GPUs that lack half precision
- **Smaller variants**: `base` and `tiny` stay available for experiments and can learn from Large by distillation
- **Drop in ONNX**: the export matches what RiShade loads, with dynamic width and height
- **Runs on any DirectX 12 GPU** through DirectML in RiShade, NVIDIA, AMD and Intel alike. RiShade runs it on the GPU the game uses, picks the full precision copy where half precision is missing, lowers the depth size on slower GPUs and falls back to its built in estimate when even the smallest size is too slow
- **Live benchmark** against Depth Anything V2 Small, refreshed by a scheduled workflow

## Usage

<details>
  <summary><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/capture-dark.svg"><img src="assets/readme/capture-light.svg" alt="Capturing training data" width="82%" height="40" align="left"></picture><picture><img src="assets/readme/spacer.svg" alt="" width="1" height="40" align="middle"></picture></summary>

You need Python 3.11 and Roblox Studio on Windows.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Start the capture server and leave it running:

```bash
.venv\Scripts\python tools\capture_server.py
```

Copy `studio/RiDepthCapture.lua` into `%LOCALAPPDATA%\Roblox\Plugins`, restart Studio and open a place. Click **Plugins**, **RiDepth**, **Capture** and allow HTTP requests when Studio asks. The camera visits 400 random spots and every sample lands in `data/`.

Keep Studio in front while it runs. The server pauses on its own when another window covers it, and the plugin hides the place's own UI so it never ends up in the pictures. Capture in many different places, variety matters more than sample count.

To add real games, turn RiShade off, record while you play and label the frames:

```bash
.venv\Scripts\python tools\record_gameplay.py --output gameplay\<game name>
.venv\Scripts\python tools\label_gameplay.py
```

More detail is in [docs/capture.md](docs/capture.md).

</details>

<details>
  <summary><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/training-dark.svg"><img src="assets/readme/training-light.svg" alt="Training a model" width="82%" height="40" align="left"></picture><picture><img src="assets/readme/spacer.svg" alt="" width="1" height="40" align="middle"></picture></summary>

Training needs an NVIDIA GPU. Install the CUDA build of PyTorch first:

```bash
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Fine tune RiDepth Large. A smaller variant can learn from it afterwards:

```bash
.venv\Scripts\python train.py --variant large --runs runs\large --epochs 20 --batch 8 --learning-rate 5e-5
.venv\Scripts\python train.py --variant tiny --teacher runs\large\best.pt --teacher-variant large --runs runs\tiny --epochs 80 --batch 16
```

| Option | What it does |
| --- | --- |
| `--variant` | `large`, `base` or `tiny` |
| `--teacher` | Weights of a trained model to learn from |
| `--teacher-variant` | Variant of the teacher, `base` by default |
| `--size` | Training height and width, 294 518 by default. Large needs multiples of 14 |
| `--epochs` | Passes over the data, 60 by default |
| `--batch` | Images per step, lower it if the GPU runs out of memory |
| `--resume` | Continue from saved weights |
| `--gameplay` | Folder of labelled gameplay frames, `gameplay` by default |
| `--studio-share` | Share of each epoch drawn from Studio samples, 0.25 by default |

The best weights are kept in `best.pt` inside the runs folder. More detail is in [docs/training.md](docs/training.md).

</details>

<details>
  <summary><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/export-dark.svg"><img src="assets/readme/export-light.svg" alt="Exporting for RiShade" width="82%" height="40" align="left"></picture><picture><img src="assets/readme/spacer.svg" alt="" width="1" height="40" align="middle"></picture></summary>

```bash
.venv\Scripts\python export.py --variant large --weights runs\large\best.pt --output exports\ridepth-large.onnx --fp16
.venv\Scripts\python export.py --variant large --weights runs\large\best.pt --output exports\ridepth-large-fp32.onnx
```

The exporter checks the model at 266 and 518 pixels before it finishes. Publish both files in a release tagged `large-` and a number, and RiShade picks up the newest one on its own.

| | |
| --- | --- |
| Input | `pixel_values`, `[1, 3, height, width]`, ImageNet normalised RGB |
| Outputs | `predicted_depth` `[1, height, width]` (higher is closer, sky is 0) and `normals` `[1, 3, height, width]` in camera space |
| Dynamic axes | `batch_size`, `height`, `width` |

The full format is described in [docs/model-format.md](docs/model-format.md).

</details>

<details>
  <summary><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/benchmark-dark.svg"><img src="assets/readme/benchmark-light.svg" alt="Benchmarking" width="82%" height="40" align="left"></picture><picture><img src="assets/readme/spacer.svg" alt="" width="1" height="40" align="middle"></picture></summary>

Time any ONNX depth model on your own GPU through DirectML, the same way RiShade runs it:

```bash
.venv\Scripts\python -m pip install onnxruntime-directml
.venv\Scripts\python tools\benchmark.py exports\ridepth-large.onnx --sizes 266 392 518
```

Rebuild the README chart locally:

```bash
.venv\Scripts\python -m tools.render_benchmark --device "my PC"
```

Model speed does not depend on trained weights, so the chart uses freshly built models and never needs the training data.

</details>

<details>
  <summary><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/layout-dark.svg"><img src="assets/readme/layout-light.svg" alt="Project layout" width="82%" height="40" align="left"></picture><picture><img src="assets/readme/spacer.svg" alt="" width="1" height="40" align="middle"></picture></summary>

| Path | What it is |
| --- | --- |
| `studio/RiDepthCapture.lua` | Studio plugin that moves the camera and ray traces depth and normals |
| `tools/capture_server.py` | Local server on `127.0.0.1:8765` that takes the screenshots and saves samples |
| `ridepth/model.py` | `RiDepthLarge` (Depth Anything V2 Small with a normals head), `RiDepthNet` (MobileNetV3 Large) and `RiDepthTiny` (MobileNetV3 Small) |
| `ridepth/losses.py` | Scale invariant depth, gradient and normal losses |
| `train.py` | Training with optional distillation |
| `export.py` | ONNX export with shape checks |
| `tools/record_gameplay.py` | Saves Roblox frames while you play |
| `tools/label_gameplay.py` | Labels gameplay frames with Depth Anything V2 Large and masks the interface |
| `tools/evaluate.py` | Scores checkpoints on Studio and on unseen games |
| `tools/benchmark.py` | DirectML or CPU timing |
| `tools/render_benchmark.py` | Builds the chart at the top of this page |

</details>

## Built With

[![Python][shield-python]][link-python] [![PyTorch][shield-pytorch]][link-pytorch] [![ONNX][shield-onnx]][link-onnx] [![Luau][shield-luau]][link-luau]

## License

[![License][shield-license]][link-license]

RiDepth is released under the [MIT License](LICENSE). RiDepth Large starts from Depth Anything V2 Small (Apache 2.0), downloaded from Hugging Face at a pinned revision. The smaller encoders start from torchvision's ImageNet MobileNetV3 weights (BSD 3 Clause). The benchmark downloads Depth Anything V2 Small (Apache 2.0) from Hugging Face to compare against, and the gameplay labeller downloads Depth Anything V2 Large (CC BY NC 4.0) as its teacher. Neither is included in this repository. The Windows icon is from [Voidstrap](https://github.com/KloBraticc/Voidstrap).

[shield-license]:      https://img.shields.io/github/license/KloBraticc/RiDepth?style=for-the-badge
[shield-license-flat]: https://img.shields.io/github/license/KloBraticc/RiDepth?style=flat&color=4c1
[shield-repo-stars]:   https://img.shields.io/github/stars/KloBraticc/RiDepth?style=flat&color=ffd700&label=stars
[shield-last-commit]:  https://img.shields.io/github/last-commit/KloBraticc/RiDepth?style=flat&color=8a2be2
[shield-python]:       https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[shield-pytorch]:      https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[shield-onnx]:         https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white
[shield-luau]:         https://img.shields.io/badge/Luau-00A2FF?style=for-the-badge

[link-license]:        https://github.com/KloBraticc/RiDepth/blob/main/LICENSE
[repo-stargazers]:     https://github.com/KloBraticc/RiDepth/stargazers
[repo-commits]:        https://github.com/KloBraticc/RiDepth/commits/main
[link-python]:         https://www.python.org/
[link-pytorch]:        https://pytorch.org/
[link-onnx]:           https://onnx.ai/
[link-luau]:           https://luau.org/
