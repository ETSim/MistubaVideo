# Source plugins

The renderer owns cameras, lighting, materials, frame rendering, overlays and encoding. Source plugins own
input formats, dependencies and project conventions. The core includes manifest and USD inputs and has no HDF5
or simulator dependency. Scalar fields are optional.

## Register a source

```toml
[project.entry-points."mitsuba_video.sources"]
my_simulation = "my_package.video:SimulationSource"

[project.entry-points."mitsuba_video.fixtures"]
my_simulation = "my_package.examples:write_example"
```

The source class implements `probe(path)` and `open(path, **options)`. An opened source supplies bodies,
frames, poses, a title, an up axis and units. `fields={}` and `primary_field=""` describe an animation without
scalar data. For scalar fields, provide `FieldSpec` metadata and `field(frame, body_id, name)`, returning a
normalized image or `None` when no image exists. Geometry-only sources do not need a field reader.

Mesh-local arrays and a material provider belong to each `Body`. Poses return `[N,3]` positions and `[N,4]`
quaternions in `(w,x,y,z)` order, indexed by body ID. A material provider builds base and affected appearances;
`build_stack` can supply one, two or four stages per channel. Field values have no prescribed physical meaning.

`mitsuba-video sources` lists installed adapters. `--source NAME` selects one explicitly. `-O KEY=VALUE` passes
options to the adapter. A plugin may provide its own console command with defaults and specialized options by
calling `build_app(default_source=..., default_look=..., fixture_kind=...)`.

## TextureFriction migration

In 0.4, the HDF5 reader, recording conventions, material-atlas adapter, viewer defaults, example recording writer
and `tf-render` command moved to the TextureFriction repository:
`scripts/tf/src/tfdev/render/source/`. They are registered by the `texturefriction-devtools` distribution.

From the TextureFriction repository, install its vendored general renderer and the plugin together:

```bash
python -m pip install -r tools/render-requirements.txt
tf-render inspect path/to/simulation_run
tf-render render path/to/simulation_run --look worn -q preview
```

`tf render setup` creates the dedicated rendering environment. The core's old `[texturefriction]` extra and
`mitsuba_video.sources.texturefriction` module are removed. Code importing the adapter should import
`tfdev.render.source`, or use `open_source(path, "texturefriction")` through the registered interface.

Recording-specific `--scale`, `--assets`, `--scratches`, `--burnish`, `--wear-edges`, `--height-gate`, `--edge-wear`
and `--wear-tint` options belong to `tf-render`. That command translates them into plugin options and keeps its
heatmap default. The general CLI defaults to `plain`; use `--look material` for a field-driven transition.
Older `--look worn` and manifest `worn` keys remain accepted.

The renderer's tests run independently. Recording, atlas-selection and export-parity tests live beside the
plugin in `scripts/tf/tests/mitsuba_plugin/` within TextureFriction.
