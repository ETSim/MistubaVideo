"""Root and subdirectory installs describe the same public Python package."""

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = pytest.importorskip("tomli")


def test_root_install_matches_distribution():
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / "build_support").is_dir():
        pytest.skip("Repository-root packaging check")
    top = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package = tomllib.loads((root / "packages/mitsuba_video/pyproject.toml").read_text(encoding="utf-8"))
    for key in ("name", "version", "requires-python", "dependencies", "scripts", "optional-dependencies", "entry-points"):
        assert top["project"][key] == package["project"][key], key
    assert top["tool"]["setuptools"]["packages"]["find"]["where"] == ["packages/mitsuba_video/src"]
