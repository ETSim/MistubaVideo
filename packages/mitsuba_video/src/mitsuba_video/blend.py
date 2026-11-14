"""Public scalar-field material blend API (CPU and optional CUDA)."""

from __future__ import annotations

from .field_blend import (  # noqa: F401
    BLEND_DEVICES,
    LOCALITY_EPS,
    WEAR_EPS,
    WEAR_TINT_RGB,
    BlendParams,
    MapSet,
    VariantStack,
    blend_maps,
    blend_stack,
    blur_wear,
    edge_field,
    gaussian_blur,
    height_blend_t,
    hermite01,
    hermite_ramp,
    hermite_tangents,
    locality,
    resolve_device,
    scratch_field,
    slerp4_normals,
    slerp_normals,
    wear_blend_weights4,
    wear_remap_t,
)

from .field_blend_gpu import DeviceMaps, blend_device  # noqa: F401
