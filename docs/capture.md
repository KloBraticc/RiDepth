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

## Safety

The server only listens on `127.0.0.1`, so nothing outside your PC can reach it. It only accepts sample names made of letters, digits and underscores, rejects oversized requests, and writes nothing outside the data folder.
