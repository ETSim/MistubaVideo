# mitsuba-video

Render 3D frames and videos with Mitsuba 3. Use it for animations, camera sweeps, material comparisons or
simulation output. Scalar fields on UV-mapped surfaces are optional: temperature, damage, wear, coverage and
other quantities can drive a heatmap or a material transition.

## Install

```bash
python -m pip install -e packages/mitsuba_video
# Optional USD and Jupyter support:
python -m pip install -e "packages/mitsuba_video[usd,notebook]"
mitsuba-video doctor
```

The core package has no HDF5 or simulator dependency. Python 3.10 or newer is required. FFmpeg is needed to
encode videos; individual PNG and EXR frames can be rendered without it.

## Render an animation

```bash
mitsuba-video fixture output/cube --frames 24
mitsuba-video render output/cube -q preview --no-encode
mitsuba-video encode output/cube/render_plain --fps 40 --hold 0
```

The default fixture is a cube following a prescribed ballistic arc. It has geometry and poses with no scalar
fields. The default look is `plain`. `beauty/` contains the 3D PNGs; `frames/` contains annotated images for video
encoding. `--keep-exr` also saves linear HDR frames.

Use `--frames 0:120:2` to select input frames, `--camera fixed|track|orbit` to choose a camera path, and `--azimuth`
and `--elevation` to adjust the view. `--resume` reuses completed frames when settings match; keep input assets
unchanged between runs. `mitsuba-video tui` guides setup in the terminal.

## Add an optional scalar field

```bash
mitsuba-video fixture output/temperature --kind temperature
mitsuba-video render output/temperature --field temperature --look heatmap -q preview
mitsuba-video fixture output/damage --kind damage
mitsuba-video render output/damage --field damage --look material -q preview
```

`plain` ignores fields, `heatmap` overlays the selected quantity, and `material` blends the base and affected
material appearances. The older `worn` look remains an alias for field-driven appearance. A scalar field is a
normalized image in a body's UV coordinates; row zero is the top of the image. Geometry without fields does not
need an atlas. Display labels and thresholds belong to the source, so the renderer does not assign physical
meaning to a field.

## Write input from Python

```python
import numpy as np
from mitsuba_video.sources.manifest.writer import ManifestWriter

writer = ManifestWriter("output/run", name="My animation")
writer.add_body(0, "object", vertices, normals, uvs, faces,
                material={"base": {"albedo": [0.2, 0.4, 0.6], "roughness": 0.5}})
for t, position, orientation in frames:
    writer.add_frame(t, position, orientation)  # arrays [N,3] and [N,4], quaternion order (w,x,y,z)
writer.close()
```

To include a field, call `writer.add_field("temperature", label="temperature, normalized")`, then pass
`fields={"temperature": {0: image}}` to `add_frame`. Use `material={"base": {...}, "affected": {...}}` for a
field-driven change. The older `worn` material key is still accepted. Material channels include albedo, roughness,
normal, height, metallic and an optional AO texture. Metallic can be a constant or image. AO multiplies albedo in
linear space as an artistic approximation. The path tracer already calculates light transport.

Manifests refer to OBJ meshes and inline or MTL materials. The format is defined by
`src/mitsuba_video/sources/manifest/schema/manifest-v1.schema.json`. `mitsuba-video validate` checks the manifest;
`inspect` reports bodies, motion and available fields. `export manifest SOURCE OUT` converts an installed source
into this portable format, including fieldless animations.

## Materials and geometry

`--shading pbr|diffuse|metallic|ao` chooses surface shading independently of the field display. PBR uses the
material's maps and values. Diffuse removes the specular lobe, metallic forces a metallic surface for inspection,
and AO shows geometric hemisphere occlusion within 20% of the scene bounding diagonal.

`--face-normals` produces faceted meshes. `--displacement-scale` defaults to zero and offsets existing vertices
using the base height map: `scale * (height - 0.5)` in world units. It does not subdivide or animate displacement;
use a dense mesh for silhouette detail.

The field material blend supports one, two or four stages per channel. Scalar remapping, height shaping and
spherical normal interpolation are reusable across source formats. Legacy wear-named blend parameters retain
their serialized names for compatibility.

## Backends and performance

`--variant auto` tests CUDA, Metal, LLVM and scalar backends in that order. Explicit `cuda`, `metal`, `cpu` and
`scalar` choices are available. `--blend-device` selects scalar-field material processing separately. CUDA OptiX
`--denoise` uses albedo and normal guides; inspect fine detail after denoising.

Quality presets are `preview` (960x540, 16 samples), `draft` (1280x720, 64) and `paper` (1920x1080, 256).
Override with `--res` and `--spp`. The scene is reused across frames; unchanged poses and fields skip redundant
updates. CPU field blending updates changed tiles where supported. `render_config.json` records the actual
backend and per-step timings.

Blender Cycles and EEVEE are optional alternate backends (`--backend cycles|eevee`); set `--blender` or
`MITSUBA_VIDEO_BLENDER` to select an executable. The scene package includes an animated `.blend` file. Surface
shading overrides and faceted normals currently require the Mitsuba backend. `--displace` retains the subdivided,
per-frame height-map path alongside the simpler static `--displacement-scale` option.

`--annotate labels.json --resume --recompose` rebuilds annotations and panels from saved beauty frames. Frame
reading and PNG/composite writing can overlap path tracing; timings distinguish background writing from the
render loop.

## Source plugins

Manifest and USD are built in. Other projects provide adapters in their own packages and register entry points:

```toml
[project.entry-points."mitsuba_video.sources"]
my_simulation = "my_package.video:SimulationSource"
```

Implement the `Source` protocol in `source.py`: bodies, frames, poses, metadata and optional fields. Set
`fields={}` and `primary_field=""` for geometry-only sources; the renderer will not call a field reader.
Each body supplies a material provider. A provider can return base/affected maps or implement `build_stack`
for multiple stages. Optional fixture writers register under `mitsuba_video.fixtures`.

Plugins own file-format dependencies, simulator conventions, presets and specialized CLI options. The renderer
loads installed plugins through package metadata without searching sibling checkouts. See the repository's
`docs/plugins.md` for the interface and migration guide.

## Tests and examples

```bash
python -m pip install -e "packages/mitsuba_video[test]"
python -m pytest packages/mitsuba_video/tests -q
python -m ruff check packages/mitsuba_video examples
python -m build packages/mitsuba_video
```

The repository notebook, `notebooks/getting_started.ipynb`, covers plain animation, multiple views, encoding,
faceted normals, displacement, USD and optional fields. Adapter-specific tests live with their plugins.
