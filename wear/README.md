# tf-render: Mitsuba 3 wear videos

`tf-render` turns a TextureFriction `--record` run into a path-traced wear video for papers and talks.

- **Geometry:** every body uses its recorded pose (`T(x) * R(q) * S(scale)`).
- **Material:** each frame gets its own worn material, from a NumPy port of the simulator's `WearAtlasBake` flat→worn blend.
- **Side panel:** a UV wear-atlas view and a live worn-area chart.

```bash
pip install -e .            # or: python tools/render.py ... from the TextureWear repo, no install
tf-render inspect simulation_results/run/simulation_<ts>
tf-render preview simulation_results/run/simulation_<ts>                      # 960x540, 16 spp, every 8th frame
tf-render render  simulation_results/run/simulation_<ts> -q paper --camera orbit \
    --azimuth -80 --orbit-degrees 60 --zoom 1.3 --fps 24 --title "Texture-space wear on concrete"
tf-render encode  simulation_results/run/simulation_<ts>/render_heatmap --fps 15 --hold 4   # no re-render
tf-render render  ... --resume                                                # continue an interrupted render
```

| Command | What it does |
|---|---|
| `render` | Renders every selected frame, then composites the panel and chart and encodes `<stem>.mp4`. `--quality preview\|draft\|paper` sets resolution and spp; `--res`/`--spp` override the preset. |
| `preview` | Quick check of framing and look before a long render. |
| `inspect` | Bodies, scale, materials, frames, resets, and wear coverage at the first and last frame. |
| `encode` | Re-encodes the frames with a new fps, title card, hold or GIF. No path tracing. |
| `fixture` | Writes a tiny synthetic recording, used by the tests and the CI smoke render. |
| `version` | Prints the tf-render and Mitsuba versions and the available RGB variants. |

## Outputs

Each render writes `beauty/` (the 3D render), `frames/` (the frames that get encoded), `exr/` (with `--keep-exr`) and the video into its output folder. It also writes:

- **`worn_area.csv`:** worn texels and worn area per frame. The chart is drawn from this file.
- **`render_config.json`:** the settings signature that `--resume` checks, the Mitsuba variant, and timings.

## Variants and assets

- **Mitsuba variant:** `cuda_ad_rgb` is used when a GPU is available, otherwise `llvm_ad_rgb`, otherwise `scalar_rgb` (override with `--variant`).
- **`.mtl` textures:** looked up next to the OBJ path stored in the HDF5. If that path doesn't exist here, its `resources/...` suffix is searched under `--assets` roots, then `$TF_ASSET_ROOT`, then every folder above the recording, then the working directory.

## Tests

`pytest` runs on a synthetic recording. The render smoke test is skipped when Mitsuba is not installed.

This package lives in `ETSim/TextureWear` under `tools/mitsuba_render`; `ETSim/MistubaVideo` carries a synced copy under `wear/`. The [full guide](https://github.com/ETSim/TextureWear/blob/wear-friction-law/docs/RENDERING.md) covers conventions and known limits.
