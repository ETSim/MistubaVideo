"""Runs inside Blender: build the scene of a ``mitsuba_video.blender`` package, save it, render the frames.

    blender -b --factory-startup --python-exit-code 1 --python blender_scene.py -- <out>/blender/scene.json

Needs only bpy, mathutils and numpy (Blender's bundled Python has no PIL or h5py), so it imports nothing from the
package. Source coordinates are kept: every object hangs under one ``simulation_root`` empty that turns the source's up
axis into Blender's +Z. Each frame is rendered to linear EXR and tone-mapped here with the Mitsuba path's ACES fit
(``video.tonemap``), so --exposure and the look of the beauty frames match Mitsuba renders.

Lights follow ``scene.FieldScene``: the key and rim rectangles become sun lights with the same irradiance at the scene
centre and the same angular size (so the same soft shadows), the constant fill or HDR map becomes the world, and the
ground is the same matte quad just under the lowest point of the recording.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
import traceback
import zlib

import numpy as np

PREFIX = "[mitsuba-video-blender] "


def say(message: str) -> None:
    print(PREFIX + message, flush=True)


# --- image helpers (pure numpy: unit-tested outside Blender) ---------------------------------------------------------


def tonemap(rgb: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Linear HDR -> 8-bit sRGB through the ACES fit (Narkowicz 2015): the same bytes as ``video.tonemap``."""
    x = np.maximum(np.asarray(rgb, dtype=np.float32), 0.0) * np.float32(2.0**exposure)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    out = np.where(y <= 0.0031308, y * 12.92, 1.055 * np.power(y, 1.0 / 2.4) - 0.055)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def png_bytes(rgb: np.ndarray, level: int = 1) -> bytes:
    """An 8-bit RGB PNG (row 0 = top) without PIL."""
    h, w, _ = rgb.shape
    raw = np.empty((h, 1 + 3 * w), np.uint8)
    raw[:, 0] = 0  # filter type None
    raw[:, 1:] = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(h, 3 * w)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
            + chunk(b"IEND", b""))  # fmt: skip


def up_basis(axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, e1, e2) exactly as ``camera.up_basis``."""
    up = np.zeros(3)
    up[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    e1 = np.roll(up, 1)
    return up, e1, np.cross(up, e1)


def look_rotation(eye, target, up) -> np.ndarray:
    """3x3 rotation whose columns are a Blender camera's local X (right), Y (up) and Z (backwards)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= max(np.linalg.norm(f), 1e-12)
    r = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(r) < 1e-9:  # looking straight along up: any right vector will do
        r = np.cross(f, np.roll(np.asarray(up, float), 1))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], axis=1)


# --- scene building (bpy) --------------------------------------------------------------------------------------------


class Builder:
    def __init__(self, pkg: dict, folder: str):
        import bpy

        self.bpy = bpy
        self.pkg, self.folder = pkg, folder
        self.n_frames = len(pkg["frames"])
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.scene = bpy.context.scene
        lo, hi = (np.asarray(v, float) for v in pkg["bounds"])
        self.lo, self.hi = lo, hi
        self.center = 0.5 * (lo + hi)
        self.extent = max(float(np.linalg.norm(hi - lo)), 1e-6)
        self.up, self.e1, self.e2 = up_basis(pkg["up_axis"])
        self.root = bpy.data.objects.new("simulation_root", None)
        self.scene.collection.objects.link(self.root)
        self.root.rotation_mode = "XYZ"
        self.root.rotation_euler = {"y": (math.pi / 2, 0.0, 0.0), "x": (0.0, -math.pi / 2, 0.0)}.get(
            pkg["up_axis"], (0.0, 0.0, 0.0))  # source up -> Blender +Z

    def path(self, rel: str) -> str:
        return os.path.normpath(os.path.join(self.folder, rel))

    def link(self, obj):
        self.scene.collection.objects.link(obj)
        obj.parent = self.root  # the parent inverse stays identity: local coordinates are source coordinates
        return obj

    # render settings

    def setup_render(self) -> dict:
        scene, pkg = self.scene, self.pkg
        w, h = pkg["resolution"]
        scene.render.resolution_x, scene.render.resolution_y = int(w), int(h)
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.render.fps = max(1, round(pkg["fps"]))
        scene.frame_start, scene.frame_end = 1, max(1, self.n_frames)
        try:
            scene.view_settings.view_transform = "Standard"  # EXR is written linear; the tone map is ours
        except TypeError:
            pass
        scene.view_settings.exposure = 0.0
        settings = scene.render.image_settings
        settings.file_format = "OPEN_EXR"
        settings.color_mode = "RGB"
        settings.color_depth = "32"
        if hasattr(settings, "exr_codec"):
            settings.exr_codec = "ZIP" if pkg["keep_exr"] else "NONE"
        info = {"engine": pkg["engine"]}
        if pkg["engine"] == "cycles":
            scene.render.engine = "CYCLES"
            info["device"] = self._cycles_device()
            cy = scene.cycles
            cy.samples = int(pkg["spp"])
            cy.max_bounces = int(pkg["max_depth"])
            cy.seed = int(pkg["seed"])
            cy.use_denoising = bool(pkg["denoise"])
            if pkg["denoise"] and info["device"] == "OPTIX":
                cy.denoiser = "OPTIX"
        else:
            for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):  # 4.2-4.5 call it EEVEE Next; 5.0 EEVEE again
                try:
                    scene.render.engine = engine
                    break
                except TypeError:
                    continue
            ee = scene.eevee
            ee.taa_render_samples = int(pkg["spp"])
            for name, value in (("use_raytracing", True), ("use_shadows", True), ("use_gtao", True),
                                ("use_soft_shadows", True)):  # fmt: skip
                if hasattr(ee, name):
                    setattr(ee, name, value)
            info["device"] = "GPU (OpenGL/Vulkan)"
        info["engine_id"] = scene.render.engine
        return info

    def _cycles_device(self) -> str:
        """The first GPU backend with a device (OptiX, CUDA, HIP, oneAPI, Metal), else CPU.
        ``MITSUBA_VIDEO_BLENDER_DEVICE=CPU`` (or a backend name) forces the choice."""
        wanted = os.environ.get("MITSUBA_VIDEO_BLENDER_DEVICE", "").upper()
        if wanted == "CPU":
            self.scene.cycles.device = "CPU"
            return "CPU"
        try:
            prefs = self.bpy.context.preferences.addons["cycles"].preferences
        except KeyError:
            return "CPU"
        for kind in ([wanted] if wanted else []) + ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]:
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            if hasattr(prefs, "refresh_devices"):
                prefs.refresh_devices()
            else:
                prefs.get_devices()
            devices = [d for d in prefs.devices if d.type == kind]
            if devices:
                for d in prefs.devices:
                    d.use = d.type == kind
                self.scene.cycles.device = "GPU"
                return kind
        self.scene.cycles.device = "CPU"
        return "CPU"

    # world, lights, ground

    def setup_world(self) -> None:
        bpy, light = self.bpy, self.pkg["lighting"]
        world = bpy.data.worlds.new("simulation_world")
        self.scene.world = world
        world.use_nodes = True
        nt = world.node_tree
        bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
        if light.get("envmap"):
            from mathutils import Matrix

            env = nt.nodes.new("ShaderNodeTexEnvironment")
            env.image = bpy.data.images.load(light["envmap"], check_existing=True)
            nt.links.new(env.outputs["Color"], bg.inputs["Color"])
            # Mitsuba looks a world direction s (source axes, its envmap y-up) up at u = atan2(s.x, -s.z) / 2pi,
            # v = acos(s.y) / pi; Blender's equirectangular lookup of v = Rz(90) Rx(90) s gives the same texel. The
            # world sees Blender axes b = root s, so the lookup vector is Rz(90) Rx(90) root^-1 b.
            to_mitsuba = (Matrix.Rotation(math.pi / 2, 3, "Z") @ Matrix.Rotation(math.pi / 2, 3, "X")
                          @ self.root.rotation_euler.to_matrix().inverted())  # fmt: skip
            coords = nt.nodes.new("ShaderNodeTexCoord")
            mapping = nt.nodes.new("ShaderNodeMapping")
            mapping.vector_type = "VECTOR"
            mapping.inputs["Rotation"].default_value = to_mitsuba.to_euler("XYZ")
            nt.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
            nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
            bg.inputs["Strength"].default_value = float(light.get("envmap_scale", 1.0))
        else:
            fill = float(light.get("fill", 0.0))
            bg.inputs["Color"].default_value = (fill, fill, fill, 1.0)
            bg.inputs["Strength"].default_value = 1.0

        az0 = float(self.pkg["azimuth"])
        if light.get("key_strength", 0) > 0:
            self._sun("key_light", az0 + 50.0, 55.0, 2.5 * self.extent, 0.9 * self.extent, light["key_strength"])
        if light.get("rim_strength", 0) > 0:
            self._sun("rim_light", az0 + 200.0, 35.0, 2.5 * self.extent, 0.6 * self.extent, light["rim_strength"])
        if light.get("ground", True):
            self._ground(float(light.get("ground_albedo", 0.32)))

    def _sun(self, name: str, az_deg: float, el_deg: float, dist: float, size: float, strength: float) -> None:
        """Mitsuba's rectangle of radiance ``strength d^2 / size^2`` gives irradiance ``strength`` at the centre; a
        sun of that strength (W/m^2) and the rectangle's angular size lights the scene the same at any scale."""
        from mathutils import Matrix, Vector

        az, el = math.radians(az_deg), math.radians(el_deg)
        d = math.cos(el) * (math.cos(az) * self.e1 + math.sin(az) * self.e2) + math.sin(el) * self.up
        data = self.bpy.data.lights.new(name, "SUN")
        data.energy = float(strength)
        data.angle = 2.0 * math.atan(0.5 * size / dist)
        if hasattr(data, "use_shadow_jitter"):  # EEVEE (4.2+): soft shadows from the sun's angle, over the samples
            data.use_shadow_jitter = True
        obj = self.link(self.bpy.data.objects.new(name, data))
        rot = Vector(d.tolist()).to_track_quat("Z", "Y")  # a sun shines along its local -Z: +Z points at the light
        obj.matrix_basis = Matrix.Translation((self.center + dist * d).tolist()) @ rot.to_matrix().to_4x4()

    def _ground(self, albedo: float) -> None:
        bpy = self.bpy
        axis = {"x": 0, "y": 1, "z": 2}[self.pkg["up_axis"]]
        gp = self.center.copy()
        gp[axis] = self.lo[axis] - 2e-3 * self.extent
        s = 40.0 * self.extent
        corners = [gp - s * self.e1 - s * self.e2, gp + s * self.e1 - s * self.e2, gp + s * self.e1 + s * self.e2,
                   gp - s * self.e1 + s * self.e2]  # counter-clockwise about up: the face normal is +up
        mesh = bpy.data.meshes.new("ground")
        mesh.from_pydata([c.tolist() for c in corners], [], [(0, 1, 2, 3)])
        mat = bpy.data.materials.new("ground")
        mat.use_nodes = True
        nt = mat.node_tree
        for node in list(nt.nodes):
            if node.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(node)
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (albedo, albedo, albedo, 1.0)
        output = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        nt.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
        mesh.materials.append(mat)
        self.link(bpy.data.objects.new("ground", mesh))

    # bodies

    def _image(self, nt, rel: str, colorspace: str, sequence: bool):
        bpy = self.bpy
        img = bpy.data.images.load(self.path(rel), check_existing=False)
        try:
            img.colorspace_settings.name = colorspace
        except TypeError:
            img.colorspace_settings.name = "Non-Color" if colorspace != "sRGB" else "sRGB"
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.interpolation = "Linear"
        node.extension = "REPEAT"
        if sequence:
            img.source = "SEQUENCE"
            user = node.image_user
            user.frame_start = 1
            user.frame_offset = 0
            user.frame_duration = self.n_frames
            user.use_auto_refresh = True
            user.use_cyclic = False
        return node

    def _material(self, body: dict):
        bpy = self.bpy
        mat = bpy.data.materials.new(f"body_{body['index']}_{body['name']}")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        tex, seq = body["textures"], body["sequence"]
        if "base_color" in tex:
            node = self._image(nt, tex["base_color"], "sRGB", seq)
            nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        if "roughness" in tex:
            node = self._image(nt, tex["roughness"], "Non-Color", seq)
            nt.links.new(node.outputs["Color"], bsdf.inputs["Roughness"])
        if "metallic" in tex:
            node = self._image(nt, tex["metallic"], "Non-Color", seq)
            nt.links.new(node.outputs["Color"], bsdf.inputs["Metallic"])
        else:
            bsdf.inputs["Metallic"].default_value = float(body["metallic"])
        if "normalmap" in tex:
            node = self._image(nt, tex["normalmap"], "Non-Color", seq)
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.space = "TANGENT"
            nmap.uv_map = "UVMap"
            nt.links.new(node.outputs["Color"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        plan = body.get("displacement")
        if plan and "height" in tex:
            self._displacement(mat, self._image(nt, tex["height"], "Non-Color", seq), plan)
        return mat

    def _displacement(self, mat, height_node, plan: dict) -> None:
        """``displace.DisplacedMesh.offsets`` as nodes: min(amplitude (h - top), 0) along the normal, in world units
        (the root has no scale), so the asperity tops stay on the collider surface and the valleys sink."""
        nt = mat.node_tree

        def math(op: str, a, b):
            node = nt.nodes.new("ShaderNodeMath")
            node.operation = op
            for i, value in enumerate((a, b)):
                if isinstance(value, (int, float)):
                    node.inputs[i].default_value = float(value)
                else:
                    nt.links.new(value, node.inputs[i])
            return node.outputs["Value"]

        h = math("SUBTRACT", height_node.outputs["Color"], plan["top"])
        offset = math("MINIMUM", math("MULTIPLY", h, plan["amplitude"]), 0.0)
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.space = "WORLD"
        disp.inputs["Midlevel"].default_value = 0.0
        disp.inputs["Scale"].default_value = 1.0
        nt.links.new(offset, disp.inputs["Height"])
        output = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        nt.links.new(disp.outputs["Displacement"], output.inputs["Displacement"])
        settings = mat if hasattr(mat, "displacement_method") else mat.cycles  # 4.2+ / older
        settings.displacement_method = "DISPLACEMENT"  # the normal map still shades the fine detail
        if hasattr(mat, "max_vertex_displacement"):  # EEVEE culling bounds
            mat.max_vertex_displacement = float(plan["amplitude"])

    def add_body(self, body: dict) -> None:
        bpy = self.bpy
        data = np.load(self.path(body["mesh"]))
        v, n, uv, f = data["vertices"], data["normals"], data["uvs"], data["faces"]
        mesh = bpy.data.meshes.new(body["name"])
        mesh.from_pydata(v.tolist(), [], f.tolist())  # loop k of face j is 3 j + k: the UV order below
        layer = mesh.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", uv[f.reshape(-1)].astype(np.float32).ravel())
        mesh.polygons.foreach_set("use_smooth", np.ones(len(f), dtype=bool))
        if hasattr(mesh, "use_auto_smooth"):  # Blender < 4.1
            mesh.use_auto_smooth = True
        nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        mesh.normals_split_custom_set_from_vertices(nn.tolist())
        mesh.materials.append(self._material(body))
        plan = body.get("displacement")
        if plan:
            self._weld(mesh, v)
        mesh.update()
        obj = self.link(bpy.data.objects.new(body["label"] or body["name"], mesh))
        if plan and plan["levels"] > 0:
            # Simple (not Catmull-Clark) splits keep the shape: the displacement needs the vertices, not smoothing.
            mod = obj.modifiers.new("displacement_subdivision", "SUBSURF")
            mod.subdivision_type = "SIMPLE"
            mod.render_levels = int(plan["levels"])
            mod.levels = min(int(plan["levels"]), 2)  # keeps the saved .blend responsive in the viewport
            if hasattr(mod, "use_custom_normals"):
                mod.use_custom_normals = True
        obj.rotation_mode = "QUATERNION"
        obj.scale = (body["scale"],) * 3
        poses = np.asarray(body["poses"], float)
        static = bool(len(poses)) and np.allclose(poses, poses[0])
        for k, row in enumerate(poses[:1] if static else poses):
            obj.location = row[:3].tolist()
            obj.rotation_quaternion = row[3:7].tolist()  # (w, x, y, z), the source's storage order
            if not static:
                obj.keyframe_insert("location", frame=k + 1)
                obj.keyframe_insert("rotation_quaternion", frame=k + 1)

    def _weld(self, mesh, vertices: np.ndarray) -> None:
        """Merge vertices that share a position (UV seams, hard edges): each then moves once, along the mean normal,
        so the displaced mesh stays closed, as ``displace.weld_groups`` does. UVs and custom normals live on the face
        corners and survive the merge."""
        import bmesh

        extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) if len(vertices) else 1.0
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(extent * 1e-7, 1e-12))
        bm.to_mesh(mesh)
        bm.free()

    def add_camera(self) -> None:
        bpy, pkg = self.bpy, self.pkg
        data = bpy.data.cameras.new("camera")
        data.sensor_fit = "HORIZONTAL"
        data.angle_x = math.radians(float(pkg["fov"]))
        data.clip_start = self.extent * 1e-3
        data.clip_end = self.extent * 1e3
        cam = self.link(bpy.data.objects.new("camera", data))
        cam.rotation_mode = "QUATERNION"
        self.scene.camera = cam
        from mathutils import Matrix

        for k, (eye, target, up) in enumerate(pkg["camera"]):
            rot = Matrix(look_rotation(eye, target, up).tolist())
            cam.location = eye
            cam.rotation_quaternion = rot.to_quaternion()
            cam.keyframe_insert("location", frame=k + 1)
            cam.keyframe_insert("rotation_quaternion", frame=k + 1)

    def save(self) -> None:
        bpy, path = self.bpy, self.pkg["blend_file"]
        self.scene.frame_set(1)
        bpy.context.preferences.filepaths.save_version = 0  # no scene.blend1 backups
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
        try:
            bpy.ops.file.make_paths_relative()
            bpy.ops.wm.save_mainfile(compress=True)
        except RuntimeError as exc:  # paths stay absolute: the .blend still opens on this machine
            say(f"could not make texture paths relative ({exc})")

    # rendering

    def read_exr(self, path: str) -> np.ndarray:
        img = self.bpy.data.images.load(path, check_existing=False)
        try:
            w, h = img.size
            px = np.empty(w * h * 4, np.float32)
            img.pixels.foreach_get(px)
        finally:
            self.bpy.data.images.remove(img)
        return px.reshape(h, w, 4)[::-1, :, :3]  # Blender rows run bottom-up

    def render(self, frames: list[int]) -> None:
        bpy, pkg = self.bpy, self.pkg
        os.makedirs(pkg["exr_dir"], exist_ok=True)
        os.makedirs(pkg["beauty_dir"], exist_ok=True)
        for k, i in enumerate(frames):
            t0 = time.time()
            self.scene.frame_set(i + 1)
            exr = os.path.join(pkg["exr_dir"], f"frame_{i:06d}.exr")
            self.scene.render.filepath = exr
            bpy.ops.render.render(write_still=True)
            ldr = tonemap(self.read_exr(exr), float(pkg["exposure"]))
            with open(os.path.join(pkg["beauty_dir"], f"frame_{i:06d}.png"), "wb") as fh:
                fh.write(png_bytes(ldr))
            say(f"frame {k + 1}/{len(frames)} {time.time() - t0:.2f} s")


def main(argv: list[str]) -> int:
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if not args:
        say("usage: blender -b --python blender_scene.py -- scene.json")
        return 2
    package = os.path.abspath(args[0])
    with open(package, encoding="utf-8") as fh:
        pkg = json.load(fh)
    import bpy

    t0 = time.time()
    b = Builder(pkg, os.path.dirname(package))
    info = b.setup_render()
    info["blender"] = bpy.app.version_string
    b.setup_world()
    for body in pkg["bodies"]:
        b.add_body(body)
    b.add_camera()
    b.save()
    info["build_seconds"] = round(time.time() - t0, 2)
    say(json.dumps(info))
    say(f"scene built in {info['build_seconds']} s: {info['engine_id']} on {info['device']}, saved {pkg['blend_file']}")
    b.render(list(pkg["render_frames"]))
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        raise
    if code:
        sys.exit(code)
