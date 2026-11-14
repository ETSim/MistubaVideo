"""Render the README cube-toss filmstrip without wear or external image assets."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from mitsuba_video.sources.manifest.cube_throw import write_cube_throw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="auto")
    parser.add_argument("--animate", action="store_true", help="Also encode the README GIF (requires FFmpeg)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "output/readme-input"
    out = root / "output/readme-render"
    write_cube_throw(source, frames=4)
    command = [sys.executable, "-m", "mitsuba_video", "render", str(source), "--out", str(out),
               "--look", "plain", "--res", "512x288", "--spp", "256", "--no-panel", "--no-chart",
               "--variant", args.variant, "--no-encode", "--face-normals", "--elevation", "18", "--azimuth", "-15"]
    if args.variant in ("auto", "cuda"):
        command.append("--denoise")
    subprocess.run(command, check=True)
    canvas = Image.new("RGB", (1024, 652), "#13202b")
    draw = ImageDraw.Draw(canvas)
    for i in range(4):
        x, y = (i % 2) * 512, (i // 2) * 326
        canvas.paste(Image.open(out / f"beauty/frame_{i:06d}.png"), (x, y))
        draw.text((x + 16, y + 302), f"Frame {i + 1}   |   t = {i * 0.2:.1f} s", fill="#e6f0f5")
    target = root / "docs/images/cube-throw.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)
    print(target)
    if args.animate:
        animation_input = root / "output/readme-animation-input"
        animation_out = root / "output/readme-animation"
        write_cube_throw(animation_input, frames=24)
        command = [sys.executable, "-m", "mitsuba_video", "render", str(animation_input),
                   "--out", str(animation_out), "--look", "plain", "--res", "480x270", "--spp", "64",
                   "--no-panel", "--no-chart", "--variant", args.variant, "--face-normals",
                   "--elevation", "18", "--azimuth", "-15", "--fps", "40", "--hold", "0", "--gif"]
        if args.variant in ("auto", "cuda"):
            command.append("--denoise")
        subprocess.run(command, check=True)
        shutil.copy2(animation_out / "render_plain_fixed.gif", root / "docs/images/cube-throw.gif")


if __name__ == "__main__":
    main()
