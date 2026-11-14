"""Material channels survive wear, export, and tile updates."""

import numpy as np
import pytest
from PIL import Image

from mitsuba_video.blend import BlendParams, VariantStack, blend_stack
from mitsuba_video.color import FieldDisplay, srgb_to_linear
from mitsuba_video.maps import NormalOptions
from mitsuba_video.pipeline import BodyTextures, Look
from mitsuba_video.sources.manifest import InlineMaterial, ManifestSource
from mitsuba_video.sources.manifest.export import export_manifest


def test_enamel_exposes_metal_under_wear(tmp_path):
    mat = InlineMaterial({"base": {"metallic": 0.0, "roughness": 0.8},
                          "worn": {"metallic": 1.0, "roughness": 0.2}}, tmp_path, "enamel")
    stack = VariantStack.pair(*mat.build((4, 4), NormalOptions(), None))
    wear = np.zeros((4, 4), np.float32)
    wear[1:3] = 1
    result = blend_stack(wear, stack, device="cpu")
    np.testing.assert_allclose(result.metallic, wear)
    np.testing.assert_allclose(result.roughness[1:3], 0.2, atol=1e-6)
    np.testing.assert_allclose(result.roughness[0], 0.8, atol=1e-6)


def test_metallic_map_and_ao_linear_multiply(tmp_path):
    Image.fromarray(np.uint8([[0, 255], [255, 0]])).save(tmp_path / "metal.png")
    Image.fromarray(np.full((2, 2), 128, np.uint8)).save(tmp_path / "ao.png")
    mat = InlineMaterial({"base": {"albedo": [0.5, 0.5, 0.5], "metallic": "metal.png", "ao": "ao.png"}},
                         tmp_path, "maps")
    base, _ = mat.build((2, 2), NormalOptions(), None)
    np.testing.assert_array_equal(base.metallic, [[0, 1], [1, 0]])
    np.testing.assert_allclose(srgb_to_linear(base.albedo), 0.5 * 128 / 255, atol=1e-6)


def test_tile_updates_preserve_metallic_maps(tmp_path):
    size = (300, 160)
    mat = InlineMaterial({"base": {"metallic": 0}, "worn": {"metallic": 1}}, tmp_path, "tiles")
    stack = VariantStack.pair(*mat.build(size, NormalOptions(), None))
    params = BlendParams()
    textures = BodyTextures(Look.worn, stack, params, FieldDisplay(), 1e-6, 1.0, "cpu")
    wear = np.zeros(size[::-1], np.float32)
    textures.update(wear)
    wear[40:80, 110:150] = 1.0
    actual = textures.update(wear)["metallic"][..., 0]
    expected = blend_stack(wear, stack, params, device="cpu").metallic
    np.testing.assert_allclose(actual, expected)


def test_export_keeps_metallic(manifest_dir, tmp_path):
    src = ManifestSource.open(manifest_dir)
    src.bodies[0].material = InlineMaterial({"base": {"metallic": 0}, "worn": {"metallic": 1}}, tmp_path, "coat")
    exported = export_manifest(src, tmp_path / "export", material_size=8, progress=False)
    other = ManifestSource.open(exported)
    base, worn = other.bodies[0].material.build((8, 8), NormalOptions(), None)
    np.testing.assert_allclose(base.metallic, 0)
    np.testing.assert_allclose(worn.metallic, 1)
    src.close()
    other.close()


def test_cuda_blend_matches_cpu_metallic_maps(tmp_path):
    # The console runner isolates CUDA and handles Mitsuba's Windows DLL teardown crash.
    import os
    import subprocess
    import sys

    from conftest import SRC

    code = r"""
import sys
from pathlib import Path
import numpy as np
import typer
from mitsuba_video import field_blend_gpu as gpu
from mitsuba_video.blend import VariantStack, blend_stack
from mitsuba_video.cli import run_main
from mitsuba_video.maps import NormalOptions
from mitsuba_video.sources.manifest import InlineMaterial

application = typer.Typer()
@application.command()
def check():
    if reason := gpu.unavailable():
        print(reason)
        raise typer.Exit(77)
    mat = InlineMaterial({"base": {"metallic": 0}, "worn": {"metallic": 1}}, Path.cwd(), "coat")
    stack = VariantStack.pair(*mat.build((16, 16), NormalOptions(), None))
    wear = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
    cpu = blend_stack(wear, stack, device="cpu")
    cuda = blend_stack(wear, stack, device="gpu")
    for name in ("albedo", "normal", "roughness", "metallic"):
        np.testing.assert_allclose(getattr(cpu, name), getattr(cuda, name), atol=2e-5)
run_main(application, "gpu-parity", [])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(SRC)})
    if result.returncode == 77:
        pytest.skip(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
