"""Build a Mitsuba scene once, then update its poses and textures for each frame.

Each body is a ``mi.Mesh``. Vertices use ``T(x) * R(q) * S(scale)``; normals rotate with the body.
Texture updates go through ``mi.traverse`` when the body's field changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .camera import CameraPose, up_basis
from .color import srgb_to_linear
from .displace import DisplacedMesh, body_budget
from .maps import MapSet
from .model import Body
from .transforms import quaternion_to_matrix

_mi = None


class Shading(str, Enum):
    pbr = "pbr"
    diffuse = "diffuse"
    metallic = "metallic"
    ao = "ao"


def _ao_integrator(distance: float):
    mi = mitsuba()

    class AmbientOcclusion(mi.SamplingIntegrator):
        def sample(self, scene, sampler, ray, medium=None, active=True):
            import drjit as dr

            si = scene.ray_intersect(ray, active=active)
            valid = si.is_valid() & active
            direction = mi.warp.square_to_cosine_hemisphere(sampler.next_2d(valid))
            shadow = si.spawn_ray(si.to_world(direction))
            shadow.maxt = distance
            visible = dr.select(scene.ray_test(shadow, active=valid), 0.0, 1.0)
            return mi.Spectrum(visible), valid, []

    mi.register_integrator("video_ao", lambda props: AmbientOcclusion(props))
    return {"type": "video_ao"}



def mitsuba():
    if _mi is None:
        raise RuntimeError("call select_variant() first")
    return _mi


def select_variant(preference: str = "auto") -> str:
    """Pick the fastest available variant: CUDA, then LLVM, then scalar (``preference`` restricts the order)."""
    global _mi
    import mitsuba as mi

    order = {
        "auto": ["cuda_ad_rgb", "metal_ad_rgb", "llvm_ad_rgb", "scalar_rgb"],
        "cuda": ["cuda_ad_rgb"],
        "metal": ["metal_ad_rgb"],
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


def _tensor(arr):
    """A texture array as ``mi.TensorXf``: numpy arrays are uploaded, device tensors (``encode_maps_device``) are
    converted in place on the device."""
    mi = mitsuba()
    if not isinstance(arr, np.ndarray):
        if isinstance(arr, mi.TensorXf):
            return arr
        if mi.variant().startswith("cuda"):
            return mi.TensorXf(arr)
        arr = np.asarray(arr)  # CUDA blend under an LLVM/scalar Mitsuba variant: other backend, go through the host
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return mi.TensorXf(arr)


def _bitmap(arr, wrap: str = "repeat") -> dict:
    mi = mitsuba()
    arr = np.ascontiguousarray(np.asarray(arr), dtype=np.float32)  # a device tensor is downloaded once, at build
    if arr.ndim == 2:
        arr = arr[..., None]
    return {"type": "bitmap", "bitmap": mi.Bitmap(arr), "raw": True, "wrap_mode": wrap, "filter_type": "bilinear"}


def _assign(params, key: str, arr: np.ndarray) -> None:
    """Set a mesh buffer using the parameter's own array type (works on scalar and JIT variants alike)."""
    params[key] = type(params[key])(np.ascontiguousarray(arr).ravel())


def encode_maps(maps: MapSet, base_color: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """MapSet -> raw texture arrays (linear base colour, roughness, [0,1]-encoded normal). ``height`` (H x W, the
    x 1, the blended height map) is not a BSDF texture: FieldScene reads it only to displace the mesh (``--displace``)."""
    out = {
        "base_color": srgb_to_linear(maps.albedo if base_color is None else base_color),
        "roughness": np.clip(maps.roughness, 0.02, 1.0)[..., None].astype(np.float32),
        "normalmap": (maps.normal * 0.5 + 0.5).astype(np.float32),
    }
    if maps.metallic is not None:
        out["metallic"] = np.clip(maps.metallic, 0.0, 1.0)[..., None].astype(np.float32)
    if maps.height is not None:
        out["height"] = np.asarray(maps.height, np.float32)[..., None].copy()
    return out


def _tensor_hw(dr, x, shape: tuple[int, int]):
    """Row-major H x W device array(s) -> ``drjit.cuda.TensorXf`` (H, W, C): a Float gives C = 1, an Array3f is
    interleaved to C = 3 (``dr.ravel`` of a vector array is x0 y0 z0 x1 ...)."""
    from drjit.cuda import Float, TensorXf

    h, w = shape
    if isinstance(x, Float):
        return TensorXf(x, (h, w, 1))
    return TensorXf(dr.ravel(x), (h, w, 3))


def encode_maps_device(dm, base_color=None) -> dict:
    """``encode_maps`` on the device for a ``wear_blend_gpu.DeviceMaps``: the same textures as
    ``encode_maps(dm.to_mapset(), base_color)`` as ``drjit.cuda.TensorXf`` (H, W, C), evaluated and never downloaded.
    ``base_color`` (Array3f, sRGB) replaces the blended albedo (the heatmap look)."""
    import drjit as dr

    c = dr.clip(dm.albedo if base_color is None else base_color, 0.0, 1.0)
    linear = dr.select(c <= 0.04045, c / 12.92, dr.power((c + 0.055) / 1.055, 2.4))
    out = {
        "base_color": _tensor_hw(dr, linear, dm.shape),
        "roughness": _tensor_hw(dr, dr.clip(dm.roughness, 0.02, 1.0), dm.shape),
        "normalmap": _tensor_hw(dr, dm.normal * 0.5 + 0.5, dm.shape),
    }
    if dm.metallic is not None:
        out["metallic"] = _tensor_hw(dr, dr.clip(dm.metallic, 0.0, 1.0), dm.shape)
    if dm.height is not None:
        out["height"] = _tensor_hw(dr, dm.height, dm.shape)
    dr.eval(*out.values())
    return out


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
        denoise_guides: bool = False,
        rfilter: str | None = None,
        rr_depth: int = 5,
        displace: float = 0.0,
        log=None,
        shading: Shading = Shading.pbr,
        face_normals: bool = False,
    ):
        """``displace`` > 0 (scene units for the full 0..1 height range) subdivides every body that has a height map
        and moves its vertices with each frame's blended height (``displace.DisplacedMesh``); 0 keeps the meshes."""
        mi = mitsuba()
        self.bodies = bodies
        self.width, self.height = width, height
        self.up_axis = up_axis
        self._denoise_guides = denoise_guides
        self._face_normals = face_normals
        self._shading = shading
        lo, hi = bounds
        center = 0.5 * (lo + hi)
        extent = max(float(np.linalg.norm(hi - lo)), 1e-6)
        up, e1, e2 = up_basis(up_axis)

        # Albedo + shading-normal AOVs feed OptiX; without them the denoiser barely helps. They come from a separate
        # AOV-only render (render()), not from wrapping the path integrator in "aov".
        self.max_depth, self.rr_depth = max_depth, rr_depth
        integrator = _ao_integrator(extent * 0.2) if shading == Shading.ao else self.integrator_dict(max_depth, rr_depth)
        self._integrators: dict[tuple[int, int], object] = {}
        self._guide_integrator = None

        scene: dict = {
            "type": "scene",
            "integrator": integrator,
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
                    # OptiX expects per-pixel independent noise: a Gaussian filter splats each sample over its
                    # neighbours, and the denoiser keeps that correlated grain as detail. Box when denoising.
                    "rfilter": {"type": rfilter or ("box" if denoise_guides else "gaussian")},
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

        if lighting.key_strength > 0 and shading != Shading.ao:  # 0 disables it, like the rim light (a camera below the scene would see it)
            scene["key_light"] = area_light(azimuth + 50.0, 55.0, 2.5 * extent, 0.9 * extent, lighting.key_strength)
        if lighting.rim_strength > 0 and shading != Shading.ao:
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
        self._displaced: dict[int, DisplacedMesh] = {}
        budget = body_budget(sum(1 for b in bodies if displace > 0.0 and textures[b.index].get("height") is not None))
        for body in bodies:
            tex = textures[body.index]
            verts, normals = (body.vertices * body.scale).astype(np.float64), body.normals.astype(np.float64)
            uvs, faces = body.uvs, body.faces
            if displace > 0.0 and tex.get("height") is not None:
                height = np.asarray(tex["height"], np.float32)
                height = height.reshape(height.shape[0], height.shape[1])
                dmesh = DisplacedMesh.build(verts, normals, uvs, faces, displace, (height.shape[1], height.shape[0]),
                                            budget)  # fmt: skip
                self._displaced[body.index] = dmesh
                verts, normals, uvs, faces = dmesh.displaced(height), dmesh.normals, dmesh.uvs, dmesh.faces
                if log:
                    log(f"displace  body {body.index}: {len(body.faces)} -> {len(faces)} tris ({dmesh.levels} "
                        f"split(s)), relief {dmesh.amplitude * 1e3:.3g} mm, tops at h={dmesh.top or 0.0:.3f}")  # fmt: skip
            material = {
                "type": "principled", "base_color": _bitmap(tex["base_color"]),
                "roughness": _bitmap(tex["roughness"]),
                "metallic": _bitmap(tex["metallic"]) if "metallic" in tex else float(metallic.get(body.index, 0)),
                "specular": 0.5,
            }
            if shading == Shading.diffuse:
                material = {"type": "diffuse", "reflectance": _bitmap(tex["base_color"])}
            elif shading == Shading.metallic:
                material["metallic"] = 1.0
            bsdf = mi.load_dict(
                {
                    "type": "twosided",
                    "material": {
                        "type": "normalmap",
                        "normalmap": _bitmap(tex["normalmap"]),
                        "bsdf": material,
                    },
                }
            )
            props = mi.Properties()
            props["bsdf"] = bsdf
            mesh = mi.Mesh(
                f"body_{body.index}",
                len(verts),
                len(faces),
                props=props,
                has_vertex_normals=not face_normals,
                has_vertex_texcoords=True,
            )
            params = mi.traverse(mesh)
            _assign(params, "faces", faces.astype(np.uint32))
            uv = uvs.astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # OBJ v-up -> Mitsuba bitmap rows (row 0 = top), as obj's flip_tex_coords
            _assign(params, "vertex_texcoords", uv)
            _assign(params, "vertex_positions", verts.astype(np.float32))
            if not face_normals:
                _assign(params, "vertex_normals", normals.astype(np.float32))
            params.update()
            scene[f"body_{body.index}"] = mesh
            self._local[body.index] = (verts, normals)
        self._poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # last pose written per body

        self.scene = mi.load_dict(scene)
        self.params = mi.traverse(self.scene)
        self._texture_keys = {}
        for body in bodies:
            prefix = f"body_{body.index}."
            keys = {}
            for k in self.params.keys():
                if not k.startswith(prefix):
                    continue
                for name in ("base_color", "roughness", "normalmap", "metallic"):
                    suffix = "reflectance" if name == "base_color" and shading == Shading.diffuse else name
                    if k.endswith(f"{suffix}.data"):
                        keys[name] = k
            self._texture_keys[body.index] = keys

    def set_poses(self, positions: np.ndarray, orientations: np.ndarray) -> int:
        """Move the bodies to the recorded poses; returns how many meshes were rewritten. A body whose pose did not
        change (every fixed body after the first frame) keeps its buffers, so the acceleration structure is not
        dirtied for it."""
        moved = 0
        for body in self.bodies:
            if body.index >= len(positions):
                continue
            pos, quat = positions[body.index], orientations[body.index]
            last = self._poses.get(body.index)
            if last is not None and np.array_equal(last[0], pos) and np.array_equal(last[1], quat):
                continue
            self._poses[body.index] = (np.array(pos, copy=True), np.array(quat, copy=True))
            moved += 1
            self._write_world(body.index)
        return moved

    def _write_world(self, index: int) -> None:
        """Rewrite a body's world-space vertices from its last pose and current local (displaced) mesh."""
        pos, quat = self._poses[index]
        rot = quaternion_to_matrix(quat)
        local_v, local_n = self._local[index]
        _assign(self.params, f"body_{index}.vertex_positions", (local_v @ rot.T + pos).astype(np.float32))
        if not self._face_normals:
            _assign(self.params, f"body_{index}.vertex_normals", (local_n @ rot.T).astype(np.float32))

    def set_textures(self, body_index: int, tex: dict) -> None:
        """Swap a body's textures: numpy arrays (``encode_maps``) or device tensors (``encode_maps_device``). A
        displaced body also moves its vertices to the new blended height."""
        for name, key in self._texture_keys[body_index].items():
            if name in tex:
                self.params[key] = _tensor(tex[name])
        dmesh = self._displaced.get(body_index)
        if dmesh is None or tex.get("height") is None:
            return
        height = np.asarray(tex["height"], np.float32)
        self._local[body_index] = (dmesh.displaced(height.reshape(height.shape[0], height.shape[1])),
                                   self._local[body_index][1])  # fmt: skip
        if body_index in self._poses:
            self._write_world(body_index)

    def set_camera(self, pose: CameraPose) -> None:
        mi = mitsuba()
        self._to_world = mi.Transform4f().look_at(
            origin=pose.eye.tolist(), target=pose.target.tolist(), up=pose.up.tolist()
        )
        self.params["sensor.to_world"] = self._to_world

    def to_sensor(self):
        """World-to-sensor transform for OptiX (sh_normal is world-space)."""
        mi = mitsuba()
        to_world = getattr(self, "_to_world", None)
        if to_world is None:
            return mi.Transform4f()
        return to_world.inverse()

    def integrator_dict(self, max_depth: int, rr_depth: int) -> dict:
        """The path integrator (Russian roulette from ``rr_depth``). The denoiser guides are not folded in: wrapping
        it in Mitsuba's "aov" integrator made every frame path trace twice (two full-size launches in Mitsuba 3.6.4,
        ~3x the cost of path + an AOV-only pass; 2026-11-06). render() takes them from :meth:`guide_integrator`."""
        return {"type": "path", "max_depth": max_depth, "rr_depth": rr_depth}

    def guide_integrator(self):
        """The AOV-only integrator for the denoiser guides (albedo, shading normal), loaded once."""
        if self._guide_integrator is None:
            self._guide_integrator = mitsuba().load_dict({"type": "aov", "aovs": "albedo:albedo,nn:sh_normal"})
        return self._guide_integrator

    def render(self, spp: int, seed: int = 0, max_depth: int | None = None,
               rr_depth: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
        """Path trace the current poses and textures. ``max_depth`` / ``rr_depth`` override the scene's integrator
        for this render (a loaded integrator per pair is cached), so a speed sweep needs no scene rebuild."""
        mi = mitsuba()
        self.params.update()
        depth, rr = max_depth or self.max_depth, rr_depth or self.rr_depth
        if (depth, rr) == (self.max_depth, self.rr_depth):
            img = mi.render(self.scene, spp=spp, seed=seed)
        else:
            key = (depth, rr)
            if key not in self._integrators:
                self._integrators[key] = mi.load_dict(self.integrator_dict(depth, rr))
            img = mi.render(self.scene, integrator=self._integrators[key], spp=spp, seed=seed)
        arr = np.asarray(img, dtype=np.float32)
        rgb = np.ascontiguousarray(arr[..., :3])
        if not self._denoise_guides:
            return rgb
        # Same sampler seed and spp as the beauty, so the guides match what the wrapped "aov" integrator gave (to
        # float rounding, ~1e-6). Layout: albedo (3), then sh_normal (3).
        guides = np.asarray(mi.render(self.scene, integrator=self.guide_integrator(), spp=spp, seed=seed),
                            dtype=np.float32)  # fmt: skip
        albedo = np.ascontiguousarray(guides[..., 0:3])
        normals = np.ascontiguousarray(guides[..., 3:6])
        return rgb, albedo, normals


WearScene = FieldScene  # former name


class Denoiser:
    """OptiX denoiser (CUDA variants only); a no-op passthrough elsewhere.

    When albedo/normal guides are available (FieldScene built with ``denoise_guides``), they are
    passed through; without them OptiX barely reduces Monte Carlo noise on these scenes.
    """

    def __init__(self, width: int, height: int, enabled: bool, use_guides: bool = True):
        self._denoiser = None
        self._use_guides = use_guides
        if not enabled:
            return
        mi = mitsuba()
        if not mi.variant().startswith("cuda"):
            return
        try:
            self._denoiser = mi.OptixDenoiser(
                input_size=[width, height],
                albedo=use_guides,
                normals=use_guides,
                temporal=False,
            )
        except Exception as exc:  # noqa: BLE001
            if use_guides:
                try:
                    self._denoiser = mi.OptixDenoiser(
                        input_size=[width, height], albedo=False, normals=False, temporal=False
                    )
                    self._use_guides = False
                    print(f"[render] OptiX with guides failed ({exc}); falling back to beauty-only")
                except Exception as exc2:  # noqa: BLE001
                    print(f"[render] OptiX denoiser unavailable ({exc2}); continuing without it")
            else:
                print(f"[render] OptiX denoiser unavailable ({exc}); continuing without it")

    @property
    def active(self) -> bool:
        return self._denoiser is not None

    @property
    def uses_guides(self) -> bool:
        return self._denoiser is not None and self._use_guides

    def __call__(
        self,
        rgb: np.ndarray,
        albedo: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        to_sensor=None,
    ) -> np.ndarray:
        if self._denoiser is None:
            return rgb
        mi = mitsuba()
        noisy = mi.TensorXf(np.ascontiguousarray(rgb, dtype=np.float32))
        if self._use_guides and albedo is not None and normals is not None:
            kwargs = {
                "albedo": mi.TensorXf(np.ascontiguousarray(albedo, dtype=np.float32)),
                "normals": mi.TensorXf(np.ascontiguousarray(normals, dtype=np.float32)),
            }
            if to_sensor is not None:
                kwargs["to_sensor"] = to_sensor
            out = self._denoiser(noisy, **kwargs)
        else:
            out = self._denoiser(noisy)
        return np.asarray(out, dtype=np.float32)
