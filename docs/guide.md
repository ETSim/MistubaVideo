# Render a video with MistubaVideo

Install `packages/mitsuba_video` from the repository root and activate your environment. The README lists the optional extras. Run `mitsuba-video doctor` to check the renderer, FFmpeg and available inputs. The [notebook](../notebooks/getting_started.ipynb) runs the workflow with inline image previews.

## Start with a cube toss

```bash
mitsuba-video fixture output/cube --kind cube-throw --frames 24
mitsuba-video inspect output/cube
mitsuba-video render output/cube --out output/cube-preview --look plain -q preview --frames 0:1 --no-encode --no-panel --no-chart
```

This creates a cube, poses it along a ballistic arc, and renders the first frame. The fixture prescribes motion; it does not solve collisions. It has no external image assets or wear maps. Inspect `output/cube-preview/beauty/frame_000000.png` before rendering the full sequence.

## Choose a camera

```bash
mitsuba-video render output/cube --out output/front --look plain --frames 12:13 --azimuth 0 --elevation 20 -q preview --no-encode
mitsuba-video render output/cube --out output/side --look plain --frames 12:13 --azimuth 90 --elevation 20 -q preview --no-encode
mitsuba-video render output/cube --out output/orbit --look plain --camera orbit --orbit-degrees 180 -q preview --no-encode
```

Fixed framing includes the full trajectory. Tracking follows a body; orbit changes the camera angle over the selected frames. `--zoom`, `--fov`, `--azimuth` and `--elevation` adjust framing. Use a separate output folder for each view. The notebook compares the same frame from several azimuths, following the approach in Mitsuba's [multi-view tutorial](https://mitsuba.readthedocs.io/en/stable/src/rendering/multi_view_rendering.html).

## Render and encode

```bash
mitsuba-video render output/cube --out output/cube-final --look plain -q draft --no-panel --no-chart --no-encode
mitsuba-video encode output/cube-final --name cube-toss --fps 40 --hold 0
```

`beauty/` contains the tone-mapped 3D PNGs. `frames/` contains annotated PNGs, and the encoder reads those. The output MP4 is `output/cube-final/cube-toss.mp4`. Changing playback speed does not require another render. `--keep-exr` also saves linear HDR output.

For interrupted renders, repeat the render command with `--resume`. Keep the inputs and rendering settings unchanged. Existing settings are checked, but changes to every source asset are not fingerprinted.

## Compare materials and geometry

Use `--shading diffuse` for a matte surface, `--shading metallic` to force metal, or `--shading ao` to inspect geometric occlusion. Default `pbr` uses the material's values. `--face-normals` uses a constant geometric normal per triangle, giving low-poly meshes a faceted appearance. It corresponds to the effect described for OBJ `face_normals=true` in the [Mitsuba shape reference](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html).

Height maps can shape field-driven material interpolation. For minimal geometric displacement, add `--displacement-scale 0.002`. Each existing vertex moves by `scale * (height - 0.5)` along its authored normal, measured in world units. The map is sampled in UV space with bilinear interpolation and clamped edges. The base geometry is displaced once, then normals are recomputed. There is no subdivision, so a coarse plane cannot show bumps between its vertices. The notebook generates a dense grid and its own height map.

## Use your own input

For USD, install the `usd` extra and read [the USD guide](usd.md). For simulator data, use `ManifestWriter` to write OBJ meshes, materials, poses and optional fields. The notebook shows the writer API without a simulator dependency. Source adapters are installed as plugins; see [the plugin guide](plugins.md). Use `--kind temperature` or
`--kind damage` to generate scalar-field examples, then choose `--look heatmap` or `--look material`. Plain
animations can omit fields entirely.

## Check speed and quality

Use preview quality to check framing. Increase samples for the final render and inspect fine detail after enabling CUDA `--denoise`. `render_config.json` stores the selected backend, settings and per-step timings. `--variant cpu` uses LLVM when available; `scalar` is a slower fallback. `--blend-device` chooses field blending independently of path tracing.

`mitsuba-video tui` guides terminal setup and prints a command you can save. The [repo skill](../.agents/skills/mitsuba-video/SKILL.md) gives Codex the package paths, input conventions and rendering workflow. It lives in `.agents/skills/` for repository discovery. You can also copy its directory into your personal Codex skills directory.
