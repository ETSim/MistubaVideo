"""Sources provide geometry, poses and optional scalar fields.

Manifest and USD inputs are built in. Other projects register source classes under
``mitsuba_video.sources`` and optional example writers under ``mitsuba_video.fixtures``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from .model import Body, FieldSpec, Frame

ENTRY_POINT_GROUP = "mitsuba_video.sources"
BUILTIN_SOURCES = {
    "manifest": "mitsuba_video.sources.manifest:ManifestSource",
    "usd": "mitsuba_video.sources.usd:UsdSource",
}


def field_value(source: Source, frame: Frame, body_index: int, name: str | None = None) -> np.ndarray | None:
    """Read a scalar field only when the source declares one."""
    return source.field(frame, body_index, name) if source.fields else None


def available_fixtures() -> dict[str, Callable[[], Callable]]:
    """Fixture writers supplied by the renderer or installed plugins."""
    found = {
        "cube-throw": lambda: _load("mitsuba_video.sources.manifest.cube_throw:write_cube_throw"),
        "manifest": lambda: _load("mitsuba_video.sources.manifest.fixture:write_synthetic_manifest"),
        "temperature": lambda: _load("mitsuba_video.sources.manifest.fixture:write_temperature_manifest"),
        "damage": lambda: _load("mitsuba_video.sources.manifest.fixture:write_damage_manifest"),
    }
    for ep in metadata.entry_points(group="mitsuba_video.fixtures"):
        found.setdefault(ep.name, ep.load)
    return found


@runtime_checkable
class Source(Protocol):
    kind: ClassVar[str]  # registry name
    path: Path  # the file that was opened
    root: Path  # directory used for default outputs (<root>/render_<look>)
    title: str  # human description (scenario, run name...)
    up_axis: str  # "x" | "y" | "z"
    fields: dict[str, FieldSpec]
    primary_field: str
    bodies: list[Body]
    frames: list[Frame]
    capabilities: frozenset[str]  # e.g. {"reset_detection"}

    @classmethod
    def probe(cls, path: Path) -> bool:
        """True when ``path`` looks like this source's input."""
        ...

    @classmethod
    def open(cls, path: Path, **options: object) -> Source: ...

    def pose(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        """(positions [N, 3], orientations [N, 4] as (w, x, y, z)), indexed by ``Body.index``."""
        ...

    def describe(self) -> list[tuple[str, str]]:
        """Header rows for ``inspect``."""
        ...

    def close(self) -> None: ...


def _load(target: str) -> type:
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def available_sources() -> dict[str, Callable[[], type]]:
    """Registered source loaders by name: entry points first, then any built-in they did not override."""
    found: dict[str, Callable[[], type]] = {}
    try:
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            found.setdefault(ep.name, ep.load)
    except Exception:  # noqa: BLE001 - broken metadata must not hide the built-ins
        pass
    for name, target in BUILTIN_SOURCES.items():
        found.setdefault(name, lambda target=target: _load(target))
    return found


def source_class(kind: str) -> type:
    loaders = available_sources()
    if kind not in loaders:
        raise ValueError(f"unknown source {kind!r}; available: {', '.join(sorted(loaders))}")
    return loaders[kind]()


def detect_kind(path: Path) -> str:
    for name, loader in available_sources().items():
        try:
            if loader().probe(Path(path)):
                return name
        except ImportError:
            continue  # optional dependency missing (a plugin dependency)
    raise ValueError(f"no source recognises {path}; pass --source explicitly")


def open_source(path: Path, kind: str = "auto", **options: object) -> Source:
    """Open ``path`` with the named source, or the first whose ``probe`` accepts it."""
    path = Path(path)
    name = detect_kind(path) if kind == "auto" else kind
    return source_class(name).open(path, **options)


def parse_options(values: list[str]) -> dict[str, str]:
    """``-O key=value`` pairs -> dict (values stay strings; sources convert)."""
    out = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep or not key:
            raise ValueError(f"source option {value!r} is not KEY=VALUE")
        out[key.strip()] = val.strip()
    return out
