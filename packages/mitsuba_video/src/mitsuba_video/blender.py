"""Blender backends: render the beauty frames with Cycles or EEVEE instead of Mitsuba.

The scene is the one the Mitsuba path builds: the same bodies, recorded poses, camera path (``camera.camera_path``),
key/rim/fill lights and ground, and the same per-frame wear-blended textures (``pipeline.BodyTextures``). This side
writes them to a package Blender reads with nothing but bpy and numpy:

    <out>/blender/scene.json              bodies, poses, camera, lights, render settings, frames to render
    <out>/blender/meshes/body_<i>.npz     vertices (mesh-local, before scale), normals, uvs (OBJ, v up), faces
    <out>/blender/textures/body_<i>/      each distinct blended texture once: <map>_<uid>.png
    <out>/blender/seq/body_<i>/           <map>_<frame>.png per Blender frame, hard links to textures/ (an image
                                          sequence, so the saved scene.blend replays the wear when scrubbed)
    <out>/blender/scene.blend             the animated scene (open it in Blender to re-light or re-shoot)

``blender_scene.py`` (run as ``blender -b --factory-startup --python blender_scene.py -- scene.json``) builds the
scene, saves scene.blend, renders each requested frame to linear EXR and tone-maps it with the Mitsuba path's ACES
fit into ``<out>/beauty/frame_NNNNNN.png``. ``pipeline.run_render`` then composites panels and chart and encodes.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tqdm import tqdm

from .camera import CameraPose, scene_bounds
from .color import linear_to_srgb
from .maps import resize_scalar
from .model import FieldSpec, Frame
from .source import Source, field_value

if TYPE_CHECKING:
    from .pipeline import RenderSettings

ENGINES = ("cycles", "eevee")
SCRIPT = Path(__file__).with_name("blender_scene.py")
PACKAGE_VERSION = 1
PNG = {"compress_level": 1}
# Lines the Blender script prints for this side (everything else is Blender's own log).
PROGRESS_RE = re.compile(r"^\[mitsuba-video-blender\] frame (\d+)/(\d+) ([0-9.]+) s")
MESSAGE_PREFIX = "[mitsuba-video-blender] "


# --- finding Blender -----------------------------------------------------------------------------------------------


def _version_key(path: str) -> tuple[int, ...]:
    match = re.search(r"Blender[ _-]?(\d+)\.(\d+)", path)
    return tuple(int(g) for g in match.groups()) if match else (0, 0)


def blender_candidates() -> list[str]:
    """Installed Blender executables, newest first (Windows Program Files, macOS /Applications, Linux, Steam)."""
    patterns = []
    if os.name == "nt":
        for root in {os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files"}:
            patterns.append(os.path.join(root, "Blender Foundation", "Blender*", "blender.exe"))
        patterns.append(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
    elif sys.platform == "darwin":
        patterns += ["/Applications/Blender*.app/Contents/MacOS/Blender"]
    else:
        patterns += ["/opt/blender*/blender", "/usr/local/blender*/blender", "/snap/bin/blender"]
    found = [p for pattern in patterns for p in glob.glob(pattern)]
    return sorted(set(found), key=_version_key, reverse=True)


def find_blender(explicit: str | None = None) -> Path:
    """``explicit``, else ``MITSUBA_VIDEO_BLENDER`` / ``BLENDER``, else ``blender`` on PATH, else the newest install."""
    for value in (explicit, os.environ.get("MITSUBA_VIDEO_BLENDER"), os.environ.get("BLENDER")):
        if value:
            path = Path(value)
            if path.is_dir():
                path = path / ("blender.exe" if os.name == "nt" else "blender")
            if not path.is_file():
                raise FileNotFoundError(f"Blender executable {path} does not exist")
            return path
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    installed = blender_candidates()
    if installed:
        return Path(installed[0])
    raise FileNotFoundError("Blender not found: install Blender 4.2 or newer, put it on PATH, or pass --blender / "
                            "set MITSUBA_VIDEO_BLENDER to blender(.exe)")  # fmt: skip


def blender_version(exe: Path) -> str:
    """"Blender 5.0.0" (first line of ``blender --version``), or "Blender (unknown version)"."""
    try:
        res = subprocess.run([str(exe), "-b", "--factory-startup", "--version"], capture_output=True, text=True,
                             timeout=120, check=False)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return "Blender (unknown version)"
    for line in res.stdout.splitlines():
        if line.startswith("Blender "):
            return line.strip()
    return "Blender (unknown version)"


# --- the package -----------------------------------------------------------------------------------------------------


def _save_png(path: Path, arr: np.ndarray, srgb_from_linear: bool = False) -> None:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if srgb_from_linear:
        a = linear_to_srgb(a)
    Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path, **PNG)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:  # another volume, FAT, no permission: a copy costs disk only
        shutil.copy2(src, dst)


def _pose_row(pos: np.ndarray, quat: np.ndarray) -> list[float]:
    return [float(v) for v in (*pos, *quat)]


# Texture keys of ``scene.encode_maps`` and how each is stored for Blender: 8-bit sRGB / data, or 16-bit data for the
# blended height map (it drives the displacement, where 8-bit steps would show as terraces).
MAPS = {"base_color": "srgb", "roughness": "data", "normalmap": "data", "metallic": "data", "height": "data16"}


def _save_png16(path: Path, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    a = a[..., 0] if a.ndim == 3 else a
    Image.fromarray((np.clip(a, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)).save(path, **PNG)


def displacement_plan(s: RenderSettings, source: Source, body, height: np.ndarray | None, displaced_bodies: int
                      ) -> dict | None:  # fmt: skip
    """The Mitsuba path's ``--displace`` (``displace.DisplacedMesh``) for Blender: the same amplitude (scene units,
    capped at a fraction of the body's diagonal), subdivision depth (edges of about two height texels within the
    triangle budget) and asperity tops (the first frame's 99.5th height percentile). None when not displaced."""
    if s.displace_mm <= 0.0 or height is None:
        return None
    from .displace import (
        MAX_AMPLITUDE_FRACTION,
        TOP_PERCENTILE,
        body_budget,
        subdivision_levels,
    )

    h = np.asarray(height, np.float32)
    h = h[..., 0] if h.ndim == 3 else h
    unit = {"m": 1e-3, "mm": 1.0}.get(getattr(source, "length_unit", "m"), 1e-3)
    v = body.vertices.astype(np.float64) * body.scale
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) if len(v) else 0.0
    target = 2.0 * np.sqrt(body.area_per_texel(h.shape))
    return {
        "amplitude": min(s.displace_mm * unit, MAX_AMPLITUDE_FRACTION * diag),
        "top": float(np.percentile(h, TOP_PERCENTILE)),
        "levels": subdivision_levels(v, body.faces, target, body_budget(displaced_bodies)),
    }


def write_package(
    s: RenderSettings,
    source: Source,
    spec: FieldSpec,
    selected: list[Frame],
    render_frames: list[int],
    sizes: dict[int, tuple[int, int]],
    poses: list[CameraPose],
    device: str,
    out: Path,
    log: Callable[[str], None] = tqdm.write,
    progress: bool = True,
) -> Path:
    """Write ``<out>/blender`` for ``selected`` (Blender frame ``i + 1`` is ``selected[i]``) and return scene.json.
    Textures are blended for every selected frame, so the saved .blend replays the whole wear history; only frames
    in ``render_frames`` (0-based selected indices) are rendered."""
    from .pipeline import (
        BodyTextures,
        _direction_at,
        _same_field,
        material_stack,
        scene_bodies_of,
        texture_sizes,
    )

    root = out / "blender"
    for sub in ("meshes", "textures", "seq"):
        if (root / sub).is_dir():
            shutil.rmtree(root / sub)  # a package is rebuilt whole: stale uids must not leak into sequences
        (root / sub).mkdir(parents=True, exist_ok=True)

    _, scales = texture_sizes(s, source, spec, selected[0])
    bodies = scene_bodies_of(s, source)
    entries = []
    for body in bodies:
        mesh_rel = f"meshes/body_{body.index}.npz"
        np.savez(root / mesh_rel, vertices=body.vertices.astype(np.float32), normals=body.normals.astype(np.float32),
                 uvs=body.uvs.astype(np.float32), faces=body.faces.astype(np.int32))  # fmt: skip
        stack = material_stack(body, sizes[body.index], s.normals, log)
        log(f"blend     body {body.index}: {stack.describe()}")
        entries.append({
            "index": body.index, "name": body.name, "label": body.label, "mesh": mesh_rel, "scale": float(body.scale),
            "fixed": bool(body.fixed), "metallic": float(body.material.metallic), "poses": [],
            "_textures": BodyTextures(s.look, stack, s.blend, s.display, spec.threshold, scales[body.index], device),
            "_last": object(), "_uids": [], "_maps": [],
        })  # fmt: skip

    t0 = time.time()
    for frame in tqdm(selected, desc="textures", unit="frame", disable=not progress, leave=False):
        positions, orientations = source.pose(frame)
        for e in entries:
            idx = e["index"]
            e["poses"].append(_pose_row(positions[idx], orientations[idx]))
            value = field_value(source, frame, idx, spec.name)
            if e["_uids"] and _same_field(value, e["_last"]):
                e["_uids"].append(e["_uids"][-1])
                continue
            e["_last"] = value
            value_r = None if value is None else resize_scalar(value, sizes[idx])
            tex = e["_textures"].update(value_r, _direction_at(source, frame, idx, sizes[idx], s.blend))
            uid = len(e["_maps"])
            folder = root / "textures" / f"body_{idx}"
            folder.mkdir(parents=True, exist_ok=True)
            written = {}
            for key, kind in MAPS.items():
                if key == "height" and s.displace_mm <= 0.0:
                    continue  # only the displacement reads it
                if key in tex:
                    rel = f"textures/body_{idx}/{key}_{uid:05d}.png"
                    if kind == "data16":
                        _save_png16(root / rel, np.asarray(tex[key]))
                    else:
                        _save_png(root / rel, np.asarray(tex[key]), srgb_from_linear=kind == "srgb")
                    written[key] = rel
            if uid == 0:
                e["_first_height"] = None if tex.get("height") is None else np.asarray(tex["height"])
            e["_maps"].append(written)
            e["_uids"].append(uid)
    log(f"blender   textures for {len(selected)} frame(s) in {time.time() - t0:.1f} s: "
        + ", ".join(f"body {e['index']} {len(e['_maps'])} distinct" for e in entries))  # fmt: skip

    by_index = {b.index: b for b in bodies}
    displaced = sum(1 for e in entries if s.displace_mm > 0.0 and e.get("_first_height") is not None)
    for e in entries:
        maps, uids = e.pop("_maps"), e.pop("_uids")
        e.pop("_textures"), e.pop("_last")
        plan = displacement_plan(s, source, by_index[e["index"]], e.pop("_first_height", None), displaced)
        if plan is not None:
            e["displacement"] = plan
            log(f"displace  body {e['index']}: {plan['levels']} subdivision level(s), relief "
                f"{plan['amplitude'] * 1e3:.3g} mm, tops at h={plan['top']:.3f}")  # fmt: skip
        if len(maps) == 1:  # one texture set for the whole video: plain images, no sequence
            e["textures"], e["sequence"] = maps[0], False
            continue
        folder = root / "seq" / f"body_{e['index']}"
        folder.mkdir(parents=True, exist_ok=True)
        e["textures"], e["sequence"] = {}, True
        for key in maps[0]:
            for i, uid in enumerate(uids):
                _link_or_copy(root / maps[uid][key], folder / f"{key}_{i + 1:06d}.png")
            e["textures"][key] = f"seq/body_{e['index']}/{key}_{1:06d}.png"

    lo, hi = scene_bounds(source, source.frames)
    exr_dir = out / "exr" if s.keep_exr else root / "exr"
    package = {
        "version": PACKAGE_VERSION,
        "engine": s.backend,
        "title": source.title,
        "resolution": [s.width, s.height],
        "spp": s.spp,
        "max_depth": s.max_depth,
        "denoise": s.denoise,
        "seed": s.seed,
        "exposure": s.exposure,
        "fps": s.video.fps,
        "up_axis": s.camera.up,
        "fov": s.camera.fov,
        "azimuth": s.camera.azimuth,
        "bounds": [lo.tolist(), hi.tolist()],
        "lighting": asdict(s.lighting),
        "frames": [{"index": f.index, "time": f.time} for f in selected],
        "camera": [[p.eye.tolist(), p.target.tolist(), p.up.tolist()] for p in poses],
        "bodies": entries,
        "render_frames": list(render_frames),
        "exr_dir": str(exr_dir.resolve()),
        "keep_exr": s.keep_exr,
        "beauty_dir": str((out / "beauty").resolve()),
        "blend_file": str((root / "scene.blend").resolve()),
    }
    path = root / "scene.json"
    path.write_text(json.dumps(package, indent=1), encoding="utf-8")
    return path


# --- running Blender -------------------------------------------------------------------------------------------------


def run_blender(exe: Path, package: Path, total: int, log: Callable[[str], None] = tqdm.write,
                progress: bool = True) -> dict:  # fmt: skip
    """Run the scene script on ``package``; Blender's own output goes to ``blender.log`` next to it."""
    log_path = package.with_name("blender.log")
    cmd = [str(exe), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(SCRIPT), "--",
           str(package)]  # fmt: skip
    bar = tqdm(total=total, desc="blender", unit="frame", disable=not progress, dynamic_ncols=True)
    tail: list[str] = []
    info: dict = {}
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)  # fmt: skip
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            match = PROGRESS_RE.match(line)
            if match:
                bar.set_postfix(s_per_frame=match.group(3))
                bar.update()
            elif line.startswith(MESSAGE_PREFIX):
                message = line[len(MESSAGE_PREFIX):]
                if message.startswith("{"):
                    info.update(json.loads(message))
                else:
                    log(f"blender   {message}")
        code = proc.wait()
    bar.close()
    if code != 0:
        raise RuntimeError(f"Blender failed ({code}); log {log_path}:\n  " + "\n  ".join(tail[-15:]))
    return info


def render_beauty(
    s: RenderSettings,
    source: Source,
    spec: FieldSpec,
    selected: list[Frame],
    render_frames: list[int],
    sizes: dict[int, tuple[int, int]],
    poses: list[CameraPose],
    device: str,
    out: Path,
    exe: Path,
    log: Callable[[str], None] = tqdm.write,
    progress: bool = True,
) -> dict:
    """Write the package, render ``render_frames`` into ``<out>/beauty`` with Blender, return the script's report."""
    package = write_package(s, source, spec, selected, render_frames, sizes, poses, device, out, log, progress)
    log(f"blender   package {package.parent} ({len(render_frames)} frame(s) to render)")
    info = run_blender(exe, package, len(render_frames), log, progress)
    missing = [i for i in render_frames if not (out / "beauty" / f"frame_{i:06d}.png").is_file()]
    if missing:
        raise RuntimeError(f"Blender finished but {len(missing)} beauty frame(s) are missing (first {missing[0]}); "
                           f"see {package.with_name('blender.log')}")  # fmt: skip
    if not s.keep_exr:
        shutil.rmtree(package.parent / "exr", ignore_errors=True)
    return info
