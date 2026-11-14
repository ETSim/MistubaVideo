"""USD imports exercise world transforms, UV seams, units and unsupported animation."""

import numpy as np
import pytest

from mitsuba_video.source import detect_kind, open_source
from mitsuba_video.transforms import quaternion_to_matrix

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402


def make_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(2)
    stage.SetTimeCodesPerSecond(2)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    parent = UsdGeom.Xform.Define(stage, "/World")
    parent.AddTranslateOp().Set((0, 0, 100))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Quad")
    mesh.CreatePointsAttr([(-100, -100, 0), (100, -100, 0), (100, 100, 0), (-100, 100, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    translate = mesh.AddTranslateOp()
    translate.Set((0, 0, 0), 0)
    translate.Set((200, 0, 0), 2)
    rotation = mesh.AddRotateZOp()
    rotation.Set(0, 0)
    rotation.Set(90, 2)
    material = UsdShade.Material.Define(stage, "/Metal")
    shader = UsdShade.Shader.Define(stage, "/Metal/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return stage, mesh


@pytest.mark.parametrize("extension", ["usda", "usdc", "usd"])
def test_usd_transforms_units_uvs_material(tmp_path, extension):
    path = tmp_path / f"stage.{extension}"
    _stage, mesh = make_stage(path)
    assert detect_kind(path) == "usd"
    src = open_source(path)
    assert src.up_axis == "z" and len(src.frames) == 3
    assert src.frames[-1].time == 1.0
    body = src.bodies[0]
    assert len(body.faces) == 2 and body.uvs.max() == 1
    assert body.material.metallic == pytest.approx(0.9)
    pos, quat = src.pose(src.frames[-1])
    np.testing.assert_allclose(pos[0], [2, 0, 1])
    actual = body.vertices @ quaternion_to_matrix(quat[0]).T + pos[0]
    matrix = UsdGeom.XformCache(Usd.TimeCode(2)).GetLocalToWorldTransform(mesh.GetPrim())
    expected = np.array([matrix.Transform(Gf.Vec3d(*map(float, v))) for v in
                         [(-100, -100, 0), (100, -100, 0), (100, 100, 0)]]) * 0.01
    np.testing.assert_allclose(actual[:3], expected, atol=1e-6)
    src.close()


def test_usd_rejects_deformation_and_scale(tmp_path):
    path = tmp_path / "animated.usda"
    stage, mesh = make_stage(path)
    scale = mesh.AddScaleOp()
    scale.Set((1, 1, 1), 0)
    scale.Set((2, 1, 1), 2)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="animated scale"):
        open_source(path)
    scale.Set((1, 1, 1), 2)
    mesh.GetPointsAttr().Set([(0, 0, 0)] * 4, 2)
    mesh.GetPointsAttr().Set([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], 0)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="animated geometry"):
        open_source(path)


def test_usd_optional_dependency_hint(monkeypatch):
    import builtins
    from mitsuba_video.sources.usd import _usd

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "pxr":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match=r"\[usd\]"):
        _usd()


def test_usd_frame_step(tmp_path):
    path = tmp_path / "stage.usda"
    make_stage(path)
    src = open_source(path, step="2")
    assert len(src.frames) == 2
    src.close()
    with pytest.raises(ValueError, match="positive"):
        open_source(path, step="0")


def test_usdz_package(tmp_path):
    from pxr import UsdUtils

    path = tmp_path / "source.usda"
    make_stage(path)
    package = tmp_path / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(path)), str(package))
    src = open_source(package)
    assert len(src.frames) == 3 and len(src.bodies) == 1
    src.close()


@pytest.mark.parametrize("feature", ["opacity", "texture"])
def test_usd_unsupported_materials_fail_explicitly(tmp_path, feature):
    path = tmp_path / "material.usda"
    stage, _mesh = make_stage(path)
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Metal/Surface"))
    if feature == "opacity":
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.5)
    else:
        texture = UsdShade.Shader.Define(stage, "/Metal/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unsupported"):
        open_source(path)
