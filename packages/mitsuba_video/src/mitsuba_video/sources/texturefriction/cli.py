"""``tf-render``: the ``mitsuba-video`` CLI defaulting to TextureFriction recordings (``--source texturefriction``).

    tf-render render  simulation_results/run/simulation_<ts> -q paper --camera orbit --title "..."
    tf-render inspect simulation_results/run/simulation_<ts>
"""

from __future__ import annotations

from ...cli import build_app, run_main

app = build_app(default_source="texturefriction", prog="tf-render")


def main(argv: list[str] | None = None) -> None:
    run_main(app, "tf-render", argv)
