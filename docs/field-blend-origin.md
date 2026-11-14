# Field blend provenance

The scalar-field blend was developed for this renderer and the TextureFriction viewer. The CPU and CUDA kernels
are maintained here as generic interpolation utilities. The TextureFriction source adapter, recorded viewer
settings and material-atlas conventions live in TextureFriction's `scripts/tf/src/tfdev/render/source/` plugin.

Existing wear-named mathematical functions and serialized blend parameters remain available for compatibility.
They can operate on any normalized scalar field. No TextureFriction checkout, HDF5 reader or simulator package is
needed to import or use the renderer.
