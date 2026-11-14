---
name: mitsuba-video
description: Render frame sequences and videos with the MistubaVideo repository, prepare manifest or USD inputs, and troubleshoot its cameras, materials and backends.
---

# MitsubaVideo

Work from the repository root. The maintained installable renderer is `packages/mitsuba_video`; `main.py` and `src/` are the older OBJ batch application. Install the package with the extras needed for the input: `usd` for OpenUSD and `notebook` for Jupyter. Simulator adapters belong to their own plugin packages. The core has no HDF5 dependency.

Run `mitsuba-video doctor` to test a backend. `auto` probes CUDA, Metal, LLVM and scalar; `--variant cuda` requires a usable NVIDIA backend. Field blending has a separate `--blend-device` setting. `--denoise` uses guided OptiX only on CUDA.

For a self-contained animation without wear or external images:

```bash
mitsuba-video fixture output/cube --kind cube-throw --frames 24
mitsuba-video render output/cube --look plain -q preview --no-panel --no-chart --no-encode
mitsuba-video encode output/cube/render_plain --fps 40 --hold 0
```

The cube fixture is prescribed ballistic motion, not a collision simulation. When asked for a general rendering example, use it without adding a wear overlay. Use `fixture --kind temperature` or `--kind damage` for optional scalar fields. Preserve the user's choice of input and appearance.

Inspect the source, then render one frame with `--frames -1:` at low resolution before a full job. Use distinct `--out` folders for different cameras or looks. `--camera orbit` changes view over a sequence; `--azimuth` and `--elevation` select fixed views. Keep input files unchanged when using `--resume`; it checks settings but does not fingerprint all assets.

`--look plain|material|heatmap` controls the scalar field. `--shading pbr|diffuse|metallic|ao` controls surface shading. `--face-normals` discards interpolated vertex normals for a faceted appearance. `--displacement-scale` defaults to zero; it offsets existing vertices from the static base height map and does not subdivide or animate displacement. A denser mesh is required for silhouette detail.

USD supports polygon meshes, rigid transform animation and constant Preview Surface materials. Read [the USD guide](../../../docs/usd.md) in the repository root for conversion requirements. For custom simulator output, use `ManifestWriter`; quaternions are `(w,x,y,z)`, fields use image-row convention, and body IDs index the pose arrays.

Use the repository's `notebooks/getting_started.ipynb` for a complete sequence and multi-view workflow, and `docs/guide.md` for commands. Rendering from notebooks should use the console entry point in a subprocess, because Mitsuba teardown can terminate Windows Python processes. Keep generated frames and videos under ignored `output/`.

Validate code changes with `python -m pytest packages/mitsuba_video/tests -q` and `python -m ruff check packages/mitsuba_video examples`. Inspect a real output frame when changing camera, geometry or material behavior. Report the tested backend and any untested platform, and avoid general speed claims from a tiny fixture.
