# USD input

```bash
python -m pip install -e "packages/mitsuba_video[usd]"
mitsuba-video inspect examples/rigid.usda
mitsuba-video render examples/rigid.usda --look plain -q preview --no-encode
```

The importer reads visible polygon meshes and composed world transforms from `.usd`, `.usda`, `.usdc` and `.usdz` stages. Stage distances become metres; time codes become seconds using `timeCodesPerSecond`. The stage's up axis controls framing. Frames include the start time and each subsequent time code through the end time. `-O step=0.5` samples half time codes; `--frames ::2` selects every other imported frame.

Supported geometry includes triangles, convex polygons, indexed and unindexed `st` UVs, authored normals, parent transforms and static nonuniform scale. UV and normal interpolation can be constant, uniform, vertex, varying or face-varying. Convex polygons are triangulated. Meshes must use `subdivisionScheme = "none"`.

Materials use constant `UsdPreviewSurface` diffuse colour, roughness and metallic inputs. Connected textures and other shading networks need baking to a manifest. The renderer provides its own camera and lighting, including optional `--envmap lighting.exr`.

The input model describes rigid bodies. Deforming points, animated topology, scale/shear or visibility, native instances, skinning, holes, subdivision surfaces and per-face materials need conversion before import. Concave faces need triangulation. Export one polygon mesh per material. Unsupported features produce errors where detected; this importer does not provide a full USD imaging implementation.

To share the supported stage as a portable manifest:

```bash
mitsuba-video export manifest examples/rigid.usda output/usd-manifest
mitsuba-video validate output/usd-manifest
```

USD input supplies no wear images. Add per-frame fields with `ManifestWriter` when you need material transitions or heatmaps. The tutorial notebook shows the writer API.
