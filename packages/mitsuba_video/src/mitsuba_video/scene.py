"""Mitsuba 3 scene for a recording: built once, then updated in place every frame.

Bodies are ``mi.Mesh`` objects whose vertex positions/normals are rewritten per frame from the recorded pose
(``T(x) * R(q) * S(scale)``); their textures are swapped through ``mi.traverse`` only when the wear changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import CameraPose, up_basis
from .color import srgb_to_linear
from .maps import MapSet
from .model import Body
from .transforms import quaternion_to_matrix

_mi = None


def mitsuba():
    if _mi is None:
        raise RuntimeError("call select_variant() first")
    return _mi


def select_variant(preference: str = "auto") -> str:
    """Pick the fastest available variant: CUDA, then LLVM, then scalar (``preference`` restricts the order)."""
    global _mi
    import mitsuba as mi

    order = {
        "auto": ["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"],
        "cuda": ["cuda_ad_rgb"],
        "llvm": ["llvm_ad_rgb"],
        "cpu": ["llvm_ad_rgb", "scalar_rgb"],
        "scalar": ["scalar_rgb"],
    }.get(preference, [preference])
    errors = []
    for variant in order:
        if variant not in mi.variants():
            continue
        try:
            mi.set_variant(variant)
            # Variants load lazily; a tiny render proves the backend (CUDA driver / LLVM library) works.
            probe = {
                "type": "scene",
                "integrator": {"type": "path"},
                "sensor": {"type": "perspective", "film": {"type": "hdrfilm", "width": 4, "height": 4}},
            }
            mi.render(mi.load_dict(probe), spp=1)
            _mi = mi
            return variant
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{variant}: {exc}")
    raise RuntimeError("no usable Mitsuba variant: " + "; ".join(errors))


@dataclass
class LightingSettings:
    key_strength: float = 1.1  # irradiance-normalized, independent of scene scale
    fill: float = 0.18  # constant environment radiance
    rim_strength: float = 0.45
    envmap: str | None = None  # optional HDR/EXR environment map (replaces the constant fill)
    envmap_scale: float = 1.0
    ground: bool = True
    ground_albedo: float = 0.32


def _tensor(arr: np.ndarray):
    mi = mitsuba()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return mi.TensorXf(arr)


def _bitmap(arr: np.ndarray, wrap: str = "repeat") -> dict:
    mi = mitsuba()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return {"type": "bitmap", "bitmap": mi.Bitmap(arr), "raw": True, "wrap_mode": wrap, "filter_type": "bilinear"}


def encode_maps(maps: MapSet, base_color: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """MapSet -> raw texture arrays (linear base colour, roughness, [0,1]-encoded normal)."""
    return {
        "base_color": srgb_to_linear(maps.albedo if base_color is None else base_color),
        "roughness": np.clip(maps.roughness, 0.02, 1.0)[..., None].astype(np.float32),
        "normalmap": (maps.normal * 0.5 + 0.5).astype(np.float32),
    }


class FieldScene:
    def __init__(
        self,
        bodies: list[Body],
        textures: dict[int, dict[str, np.ndarray]],
        metallic: dict[int, float],
        width: int,
        height: int,
        fov: float,
        bounds: tuple[np.ndarray, np.ndarray],
        lighting: LightingSettings,
        up_axis: str = "y",
        max_depth: int = 8,
        azimuth: float = -55.0,
    ):
        mi = mitsuba()
        self.bodies = bodies
        self.width, self.height = width, height
        self.up_axis = up_axis
        lo, hi = bounds
        center = 0.5 * (lo + hi)
        extent = max(float(np.linalg.norm(hi - lo)), 1e-6)
        up, e1, e2 = up_basis(up_axis)

        scene: dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": max_depth},
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "fov_axis": "x",
                "near_clip": extent * 1e-3,
                "far_clip": extent * 1e3,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=(center + extent * (e1 + up)).tolist(), target=center.tolist(), up=up.tolist()
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": width,
                    "height": height,
                    "pixel_format": "rgb",
                    "rfilter": {"type": "gaussian"},
                },
                "sampler": {"type": "independent"},
            },
        }

        # Lights are placed relative to the camera azimuth so the key always rakes across the visible surface.
        def area_light(az_deg: float, el_deg: float, dist: float, size: float, strength: float) -> dict:
            az, el = math.radians(az_deg), math.radians(el_deg)
            d = math.cos(el) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(el) * up
            pos = center + dist * d
            radiance = strength * dist * dist / (size * size)
            return {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().look_at(origin=pos.tolist(), target=center.tolist(), up=up.tolist())
                @ mi.ScalarTransform4f().scale([size * 0.5, size * 0.5, 1.0]),
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance}},
            }

        scene["key_light"] = area_light(azimuth + 50.0, 55.0, 2.5 * extent, 0.9 * extent, lighting.key_strength)
        if lighting.rim_strength > 0:
            scene["rim_light"] = area_light(azimuth + 200.0, 35.0, 2.5 * extent, 0.6 * extent, lighting.rim_strength)
        if lighting.envmap:
            scene["environment"] = {"type": "envmap", "filename": lighting.envmap, "scale": lighting.envmap_scale}
        elif lighting.fill > 0:
            scene["environment"] = {"type": "constant", "radiance": {"type": "rgb", "value": lighting.fill}}

        if lighting.ground:
            # A large matte floor just under the lowest point gives a horizon and catches the contact shadows.
            axis = {"x": 0, "y": 1, "z": 2}[up_axis]
            ground_pos = center.copy()
            ground_pos[axis] = lo[axis] - 2e-3 * extent
            size = 40.0 * extent
            scene["ground"] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=ground_pos.tolist(), target=(ground_pos + up).tolist(), up=e1.tolist()
                )
                @ mi.ScalarTransform4f().scale([size, size, 1.0]),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": lighting.ground_albedo}},
            }

        self._local = {}
        for body in bodies:
            tex = textures[body.index]
            bsdf = mi.load_dict(
                {
                    "type": "twosided",
                    "material": {
                        "type": "normalmap",
                        "normalmap": _bitmap(tex["normalmap"]),
                        "bsdf": {
                            "type": "principled",
                            "base_color": _bitmap(tex["base_color"]),
                            "roughness": _bitmap(tex["roughness"]),
                            "metallic": float(metallic.get(body.index, 0.0)),
                            "specular": 0.5,
                        },
                    },
                }
            )
            props = mi.Properties()
            props["bsdf"] = bsdf
            mesh = mi.Mesh(
                f"body_{body.index}",
                len(body.vertices),
                len(body.faces),
                props=props,
                has_vertex_normals=True,
                has_vertex_texcoords=True,
            )
            params = mi.traverse(mesh)
            params["faces"] = mi.UInt32(body.faces.ravel())
            uv = body.uvs.astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # OBJ v-up -> Mitsuba bitmap rows (row 0 = top), as obj's flip_tex_coords
            params["vertex_texcoords"] = mi.Float(uv.ravel())
            params["vertex_positions"] = mi.Float((body.vertices * body.scale).astype(np.float32).ravel())
            params["vertex_normals"] = mi.Float(body.normals.astype(np.float32).ravel())
            params.update()
            scene[f"body_{body.index}"] = mesh
            self._local[body.index] = ((body.vertices * body.scale).astype(np.float64), body.normals.astype(np.float64))

        self.scene = mi.load_dict(scene)
        self.params = mi.traverse(self.scene)
        self._texture_keys = {}
        for body in bodies:
            prefix = f"body_{body.index}."
            keys = {}
            for k in self.params.keys():
                if not k.startswith(prefix):
                    continue
                for name in ("base_color", "roughness", "normalmap"):
                    if k.endswith(f"{name}.data"):
                        keys[name] = k
            self._texture_keys[body.index] = keys

    def set_poses(self, positions: np.ndarray, orientations: np.ndarray) -> None:
        mi = mitsuba()
        for body in self.bodies:
            if body.index >= len(positions):
                continue
            rot = quaternion_to_matrix(orientations[body.index])
            local_v, local_n = self._local[body.index]
            world_v = local_v @ rot.T + positions[body.index]
            world_n = local_n @ rot.T
            self.params[f"body_{body.index}.vertex_positions"] = mi.Float(world_v.astype(np.float32).ravel())
            self.params[f"body_{body.index}.vertex_normals"] = mi.Float(world_n.astype(np.float32).ravel())

    def set_textures(self, body_index: int, tex: dict[str, np.ndarray]) -> None:
        for name, key in self._texture_keys[body_index].items():
            if name in tex:
                self.params[key] = _tensor(tex[name])

    def set_camera(self, pose: CameraPose) -> None:
        mi = mitsuba()
        self.params["sensor.to_world"] = mi.Transform4f().look_at(
            origin=pose.eye.tolist(), target=pose.target.tolist(), up=pose.up.tolist()
        )

    def render(self, spp: int, seed: int = 0) -> np.ndarray:
        mi = mitsuba()
        self.params.update()
        img = mi.render(self.scene, spp=spp, seed=seed)
        return np.asarray(img, dtype=np.float32)[..., :3]


WearScene = FieldScene  # former name


class Denoiser:
    """OptiX denoiser (CUDA variants only); a no-op passthrough elsewhere."""

    def __init__(self, width: int, height: int, enabled: bool):
        self._denoiser = None
        if not enabled:
            return
        mi = mitsuba()
        if not mi.variant().startswith("cuda"):
            return
        try:
            self._denoiser = mi.OptixDenoiser(input_size=[width, height], albedo=False, normals=False, temporal=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[render] OptiX denoiser unavailable ({exc}); continuing without it")

    @property
    def active(self) -> bool:
        return self._denoiser is not None

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self._denoiser is None:
            return rgb
        mi = mitsuba()
        return np.asarray(self._denoiser(mi.TensorXf(np.ascontiguousarray(rgb))), dtype=np.float32)
