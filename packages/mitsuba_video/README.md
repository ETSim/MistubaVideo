# mitsuba-video: path-traced videos of fields on moving bodies

`mitsuba-video` renders rigid bodies with [Mitsuba 3](https://www.mitsuba-renderer.org/) while a per-texel scalar
*field* evolves in each body's UV atlas: wear, damage, temperature, coverage, and so on. Every frame:

- the bodies are re-posed;
- the field drives a material blend (base → affected) and an optional heatmap;
- a side panel shows the UV atlas and a covered-area chart.

The frames are then encoded to mp4.

It is project-agnostic. Inputs are pluggable **sources**:

| Source | Input | Notes |
|---|---|---|
| `manifest` | `manifest.json`: OBJ meshes, `.mtl` or inline materials, poses, field images | Any project can write one (see below) |
| `texturefriction` | a TextureFriction `--record` run (`simulation_<ts>/` or `.h5`) | Fields `wear`, `sliding`; `tf-render` is a shortcut |
| *yours* | anything | Register a class under the `mitsuba_video.sources` entry point |

## Install and run

```bash
pip install -e "packages/mitsuba_video[texturefriction]"   # drop the extra if you don't need h5py
mitsuba-video fixture demo                                   # synthetic manifest to try things on
mitsuba-video inspect demo
mitsuba-video preview demo                                   # 960x540, 16 spp, every 8th frame
mitsuba-video render  demo -q paper --camera orbit --title "Wear on a plane" --ramp 0.2
mitsuba-video encode  demo/render_heatmap --fps 15 --hold 4  # new fps/title without re-rendering
mitsuba-video render  demo --resume                          # continue an interrupted render
```

| Command | What it does |
|---|---|
| `render` | Path traces every selected frame, then encodes `<prefix>_<look>_<camera>.mp4`. `-q preview\|draft\|paper` sets resolution and spp; `--res`/`--spp` override the preset. |
| `preview` | Quick framing and look check before a long render. |
| `inspect` | Bodies, materials, frames, motion and field coverage. |
| `encode` | Re-encodes rendered frames with a new fps, title card, hold or GIF. No path tracing. |
| `fixture` | Writes a tiny synthetic input (`--kind manifest\|texturefriction`). |
| `validate` | Checks a manifest against the JSON schema and that its files exist. |
| `export manifest` | Converts any source into a manifest, e.g. a TextureFriction recording for colleagues without h5py. |
| `sources` | Lists registered sources. |
| `version` | Prints the package and Mitsuba versions. |

Common options:

- **Source:** `--source` (default `auto`), `-O KEY=VALUE` source options, `--field NAME`.
- **Look:** `--look heatmap|worn|plain`.
- **Camera:** `--camera fixed|track|orbit` with `--azimuth`, `--elevation`, `--zoom`, `--orbit-degrees`.
- **Display only** (labelled): `--ramp` (opacity fades in with the field) and `--display-max` (colour-bar range).
- **Video:** `--title`, `--subtitle`, `--hold`, `--fps`.

`--resume` refuses to continue if the settings changed. The Mitsuba variant is `cuda_ad_rgb` when available, then `llvm_ad_rgb`, then `scalar_rgb`.

## Writing a manifest from your project

```python
from mitsuba_video.sources.manifest.writer import ManifestWriter

w = ManifestWriter("out/pin_on_disc", name="pin on disc", length_unit="m")
w.add_field("wear", label="wear depth, normalized", threshold=0.002)
w.add_body(0, "disc", vertices, normals, uvs, faces, fixed=True,
           material={"base": {"albedo": [0.5, 0.5, 0.55], "roughness": 0.6},
                     "worn": {"roughness": 0.25}})
w.add_body(1, "pin", mesh="assets/pin.obj", material={"mtl": "assets/pin.mtl"})
for t, positions, quats, wear0 in my_simulation():       # positions [N,3]; quats [N,4] as (w, x, y, z)
    w.add_frame(t, positions, quats, fields={"wear": {0: wear0}})   # HxW float in [0, 1], row 0 = V = 1
w.close()                                                 # manifest.json + poses.npz + fields/*.png
```

The format is versioned JSON, described by `src/mitsuba_video/sources/manifest/schema/manifest-v1.schema.json`:

- **Bodies:** OBJ meshes, plus either an `.mtl` material (`map_Kd`, `map_Bump`, `map_Pr`, `map_disp` and their `*_worn` variants) or inline `base`/`worn` appearances (colours, values or image paths).
- **Poses:** `poses.npz` (`time[F]`, `position[F,N,3]`, `orientation[F,N,4]`) or inline per-frame entries.
- **Fields:** per-frame atlas images (`.npy` float or 8/16-bit PNG) located by `field_pattern`.

## Adding a source

Implement the `Source` protocol (`src/mitsuba_video/source.py`). It must provide:

- `probe`, `open`, `pose`, `field`, `describe`, `close`;
- the attributes `bodies`, `frames`, `fields` and `primary_field`.

Each `Body` carries a `MaterialProvider` that builds `(base, worn)` maps at a requested size. Register the class:

```toml
[project.entry-points."mitsuba_video.sources"]
mysim = "mysim_video:MySimSource"
```

## Outputs

Each render folder holds:

- `beauty/`: the 3D render.
- `frames/`: the encoded composites.
- `exr/`: linear HDR frames (only with `--keep-exr`).
- `<field>_area.csv`: covered texels and area per frame (`worn_area.csv` for TextureFriction). The chart is drawn from this file.
- `render_config.json`: the settings signature, Mitsuba variant and timings.
- The mp4 (plus a GIF with `--gif`).

## Tests

`pytest` (in this folder) covers:

- transforms and the blend math;
- the OBJ and MTL readers;
- the manifest schema and writer;
- TextureFriction → manifest parity;
- a golden comparison of the TextureFriction source against the original renderer;
- CLI and render smoke tests (skipped without Mitsuba or ffmpeg).

Origin: TextureFriction's `tools/mitsuba_render` (ETSim/TextureWear), which now installs this package pinned by commit.
