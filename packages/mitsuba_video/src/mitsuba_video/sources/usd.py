"""OpenUSD polygon meshes with rigid transform animation and constant Preview Surface materials."""

from pathlib import Path
from typing import ClassVar

import numpy as np

from ..model import Body, Frame
from ..transforms import quaternion_to_matrix
from .manifest import InlineMaterial


def _usd():
    try:
        from pxr import Gf, Usd, UsdGeom, UsdShade
    except ImportError as exc:
        raise RuntimeError('USD input needs the optional extra: pip install "mitsuba-video[usd]"') from exc
    return Gf, Usd, UsdGeom, UsdShade


def _constant(attr, time):
    value = attr.Get(time)
    for sample in attr.GetTimeSamples():
        if not np.array_equal(np.asarray(attr.Get(sample)), np.asarray(value)):
            raise ValueError(f"animated geometry/material attribute {attr.GetPath()} is unsupported; export rigid meshes")
    return value


def _mesh(mesh, time):
    points = np.asarray(_constant(mesh.GetPointsAttr(), time), dtype=np.float64)
    counts = np.asarray(_constant(mesh.GetFaceVertexCountsAttr(), time), dtype=int)
    indices = np.asarray(_constant(mesh.GetFaceVertexIndicesAttr(), time), dtype=int)
    if not len(points) or not len(counts) or np.any(counts < 3) or counts.sum() != len(indices):
        raise ValueError(f"invalid polygon topology: {mesh.GetPath()}")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError(f"invalid point indices: {mesh.GetPath()}")
    if mesh.GetHoleIndicesAttr().Get(time):
        raise ValueError(f"mesh holes are unsupported: {mesh.GetPath()}")
    if mesh.GetSubdivisionSchemeAttr().Get(time) not in (None, "none"):
        raise ValueError(f"subdivision mesh {mesh.GetPath()}: export a polygon mesh with subdivisionScheme=none")
    _, _, UsdGeom, _ = _usd()
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uv = None
    if st:
        _constant(st.GetAttr(), time)
        if st.IsIndexed():
            _constant(st.GetIndicesAttr(), time)
        uv = np.asarray(st.ComputeFlattened(time), dtype=np.float32)
    normals = mesh.GetNormalsAttr()
    normal_values = _constant(normals, time) if normals.HasValue() else None
    normal_values = None if normal_values is None else np.asarray(normal_values)
    vertices, texcoords, ns, faces = [], [], [], []
    offset = 0

    def corner(values, interpolation, face, corner_id, point_id):
        idx = {"constant": 0, "uniform": face, "vertex": point_id,
               "varying": point_id, "faceVarying": corner_id}.get(str(interpolation))
        if idx is None:
            raise ValueError(f"unsupported interpolation {interpolation}")
        return values[idx]

    for face, count in enumerate(counts):
        if count > 3:
            # Fan triangulation only preserves convex polygons. Reject concave faces explicitly.
            polygon = points[indices[offset:offset + count]]
            cross = np.cross(np.roll(polygon, -1, axis=0) - polygon,
                             np.roll(polygon, -2, axis=0) - np.roll(polygon, -1, axis=0))
            axis = np.sum(cross, axis=0)
            if np.any(cross @ axis < -1e-10):
                raise ValueError(f"concave polygon in {mesh.GetPath()}; triangulate before export")
        for k in range(1, count - 1):
            corners = [offset, offset + k, offset + k + 1]
            if mesh.GetOrientationAttr().Get(time) == "leftHanded":
                corners.reverse()
            tri = points[indices[corners]]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            start = len(vertices)
            for c in corners:
                p = indices[c]
                vertices.append(points[p])
                texcoords.append(corner(uv, st.GetInterpolation(), face, c, p) if uv is not None else [0, 0])
                ns.append(corner(normal_values, mesh.GetNormalsInterpolation(), face, c, p)
                          if normal_values is not None else normal)
            faces.append([start, start + 1, start + 2])
        offset += count
    return np.asarray(vertices), np.asarray(ns), np.float32(texcoords), np.uint32(faces)


def _material(prim, time, root):
    _, _, _, UsdShade = _usd()
    binding = UsdShade.MaterialBindingAPI(prim)
    if binding.GetMaterialBindSubsets():
        raise ValueError(f"per-face materials on {prim.GetPath()} are unsupported; split the mesh")
    material, _ = binding.ComputeBoundMaterial()
    base = {"albedo": [0.6, 0.6, 0.6], "roughness": 0.6}
    if material:
        shader, _, _ = material.ComputeSurfaceSource()
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            raise ValueError(f"{prim.GetPath()}: only UsdPreviewSurface is supported")
        for name, default in (("opacity", 1.0), ("displacement", 0.0), ("useSpecularWorkflow", 0),
                              ("emissiveColor", [0, 0, 0])):
            inp = shader.GetInput(name)
            if inp:
                value = _constant(inp.GetAttr(), time)
                if inp.HasConnectedSource() or (value is not None and not np.allclose(value, default)):
                    raise ValueError(f"USD material input {name} on {prim.GetPath()} is unsupported")
        normal = shader.GetInput("normal")
        if normal and normal.HasConnectedSource():
            raise ValueError(f"connected USD normal input on {prim.GetPath()} is unsupported; bake to a manifest")
        for input_name, output_name in (("diffuseColor", "albedo"), ("roughness", "roughness"),
                                        ("metallic", "metallic")):
            inp = shader.GetInput(input_name)
            if not inp:
                continue
            if inp.HasConnectedSource():
                raise ValueError(f"connected USD material input {input_name} is unsupported; bake to a manifest")
            value = _constant(inp.GetAttr(), time)
            if value is not None:
                base[output_name] = list(value) if input_name == "diffuseColor" else float(value)
    return InlineMaterial({"base": base}, root, str(prim.GetPath()))


class UsdSource:
    kind = "usd"
    capabilities = frozenset()
    primary_field = ""
    fields: ClassVar = {}
    length_unit = "m"

    @classmethod
    def probe(cls, path):
        return Path(path).suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}

    @classmethod
    def open(cls, path, **options):
        Gf, Usd, UsdGeom, _ = _usd()
        self = cls()
        self.path = Path(path).resolve()
        self.root, self.title = self.path.parent, self.path.stem
        self.stage = Usd.Stage.Open(str(self.path))
        if not self.stage:
            raise ValueError(f"cannot open USD stage {path}")
        self.up_axis = str(UsdGeom.GetStageUpAxis(self.stage)).lower()
        units = UsdGeom.GetStageMetersPerUnit(self.stage)
        rate = self.stage.GetTimeCodesPerSecond()
        start, end = self.stage.GetStartTimeCode(), self.stage.GetEndTimeCode()
        step = float(options.get("step", 1))
        if not np.isfinite(step) or step <= 0 or not np.isfinite([start, end, rate]).all() or rate <= 0:
            raise ValueError("USD time range, rate and step must be finite; rate and step must be positive")
        count = int(np.floor((end - start) / step)) + 1
        if not 1 <= count <= 100000:
            raise ValueError("USD frame count must be between 1 and 100000; increase -O step=...")
        times = [start + i * step for i in range(count)]
        self.frames = [Frame(str(t), i, (t - start) / rate, i) for i, t in enumerate(times)]
        self.bodies, self._poses = [], []
        for prim in self.stage.Traverse():
            if prim.IsInstance() or prim.IsInstanceProxy():
                raise ValueError(f"USD instances are unsupported: {prim.GetPath()}; deinstance before export")
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if "SkelBindingAPI" in prim.GetAppliedSchemas() or prim.HasRelationship("skel:skeleton"):
                raise ValueError(f"skinning is unsupported: {prim.GetPath()}; bake a rigid mesh")
            imageable = UsdGeom.Imageable(prim)
            visibility = [imageable.ComputeVisibility(Usd.TimeCode(t)) for t in times]
            if len(set(visibility)) > 1:
                raise ValueError(f"animated visibility is unsupported: {prim.GetPath()}")
            if visibility[0] == "invisible":
                continue
            mesh = UsdGeom.Mesh(prim)
            v, n, uv, faces = _mesh(mesh, Usd.TimeCode(start))
            poses, stretch0 = [], None
            for t in times:
                matrix = UsdGeom.XformCache(Usd.TimeCode(t)).GetLocalToWorldTransform(prim)
                raw = np.asarray(matrix)
                if not np.allclose(raw[:3, 3], 0) or not np.isclose(raw[3, 3], 1):
                    raise ValueError(f"non-affine transform: {prim.GetPath()}")
                quat = Gf.Transform(matrix).GetRotation().GetQuat()
                q = np.array([quat.GetReal(), *quat.GetImaginary()])
                rotation = quaternion_to_matrix(q)
                stretch = raw[:3, :3] @ rotation
                if abs(np.linalg.det(stretch)) < 1e-12:
                    raise ValueError(f"singular transform: {prim.GetPath()}")
                if stretch0 is None:
                    stretch0 = stretch
                elif not np.allclose(stretch, stretch0, atol=1e-6):
                    raise ValueError(f"animated scale/shear on {prim.GetPath()} is unsupported; export rigid animation")
                poses.append((raw[3, :3] * units, q))
            v = v @ stretch0 * units
            n = n @ np.linalg.inv(stretch0).T
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            if np.linalg.det(stretch0) < 0:
                faces = faces[:, ::-1].copy()
            self.bodies.append(Body(len(self.bodies), prim.GetName(), np.float32(v), np.float32(n), uv, faces,
                                    1.0, len(times) == 1 or all(np.allclose(poses[0][0], p) and
                                    np.allclose(poses[0][1], q) for p, q in poses),
                                    _material(prim, Usd.TimeCode(start), self.root)))
            self._poses.append(poses)
        if not self.bodies:
            raise ValueError("USD stage has no visible polygon meshes")
        return self

    def pose(self, frame):
        entries = [p[frame.position] for p in self._poses]
        return np.asarray([p for p, _ in entries]), np.asarray([q for _, q in entries])

    def field(self, frame, body_index, name=None):
        return None

    def describe(self):
        return [("source", f"USD: {self.path}"), ("frames", str(len(self.frames))),
                ("units", "metres"), ("up", self.up_axis)]

    def close(self):
        self.stage = None
