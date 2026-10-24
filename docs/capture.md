# Capturing training data

RiDepth learns from pairs of a Studio screenshot and the exact depth behind it. Two parts work together: a Studio plugin that moves the camera and measures depth, and a small local server that takes the screenshot and writes the files.

## How a sample is made

1. The plugin flashes the viewport magenta once. The server finds that rectangle on screen, so it knows exactly where the 3D view is.
2. For each sample the plugin picks a random open spot, 3 to 40 studs above the ground, with a random direction and field of view between 55 and 85 degrees.
3. It waits a quarter of a second for textures to settle, then asks the server to screenshot the viewport.
4. It casts one ray per cell of a 320 by 180 grid. The hit distance along the view direction is the depth, the hit normal turned into camera space is the surface direction. A ray that hits nothing is sky.
5. It posts the depth and normals, and the server saves the sample.

Parts with transparency of 0.95 or more, and parts that cannot be queried, are skipped by the rays, because they are not visible in the picture either.

## What is saved

```
data/<place id>_<time>/<sample>/
  rgb.png      640 wide screenshot
  depth.npy    float32 depth in studs, 0 means sky
  normal.npy   float16 camera space normals, shape 3 x height x width
  meta.json    field of view and grid size
```

`data/` is ignored by git, nothing you capture is committed.

## Getting good data

- Capture in many different places: obbies, cities, forests, interiors, night maps, places with glass and water.
- Turn off the grid and the view selector in Studio, anything drawn over the 3D view ends up in the pictures.
- Set Lighting the way players see the place before capturing.
- Leave the PC alone while it runs. The server pauses when Studio is not the active window and the plugin waits until it is back.

## Recording real gameplay

Studio samples are exact but come from the places you open in Studio. To teach the model what real games look like, record frames while you play and let a large teacher model label them.

1. Turn RiShade off, so its effects never end up in the frames.
2. Start the recorder and play. It only saves while the Roblox window is in front, skips frames that barely changed (relative to how bright the scene is) and skips loading screens.

```bash
.venv\Scripts\python tools\record_gameplay.py --output gameplay\<game name>
```

3. Label everything with Depth Anything V2 Large on the GPU. The first run downloads the teacher (about 640 MB, checked against a pinned hash) into the temp folder.

```bash
.venv\Scripts\python tools\label_gameplay.py
```

The labeller writes `<frame>.teacher.npy` next to each frame and a `static.png` per recording session. That mask marks the game's own interface: pixels that stay the same in most frames while the camera moves, and that sit near an edge, so plain sky is never mistaken for a HUD. Masked pixels are left out of the loss.

```
gameplay/<game>/<session>/
  000000.png           640 wide frame
  000000.teacher.npy   float16 relative disparity from the teacher
  static.png           interface mask for the session
```

Record many different games, and keep each game in its own folder. Training holds out whole games for validation, so the score shows how well the model does in games it has never seen.

`gameplay/` is ignored by git.

## Safety

The server only listens on `127.0.0.1`, so nothing outside your PC can reach it. It only accepts sample names made of letters, digits and underscores, rejects oversized requests, and writes nothing outside the data folder.
