"""Shared fixtures: one synthetic TextureFriction recording and one synthetic manifest per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # run from a checkout without `pip install -e`
    sys.path.insert(0, str(SRC))

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def tf_recording(tmp_path_factory) -> Path:
    pytest.importorskip("h5py")
    from mitsuba_video.sources.texturefriction.synthetic import write_synthetic

    return write_synthetic(tmp_path_factory.mktemp("tf") / "FrictionTexture_synthetic.h5").parent


@pytest.fixture(scope="session")
def manifest_dir(tmp_path_factory) -> Path:
    from mitsuba_video.sources.manifest.fixture import write_synthetic_manifest

    return write_synthetic_manifest(tmp_path_factory.mktemp("manifest")).parent
