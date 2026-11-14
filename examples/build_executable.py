"""Build a portable application folder and ZIP with PyInstaller."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", action="store_true", help="Bundle the optional USD reader (install [usd,freeze])")
    parser.add_argument("--ffmpeg", type=Path, help="Bundle this FFmpeg executable; otherwise use FFmpeg on PATH")
    args = parser.parse_args()
    if importlib.util.find_spec("PyInstaller") is None:
        parser.error('Install the build extra first: python -m pip install ".[freeze]"')
    if args.usd and importlib.util.find_spec("pxr") is None:
        parser.error('USD bundling needs the extra: python -m pip install ".[usd,freeze]"')
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / ("windows" if sys.platform == "win32" else sys.platform)
    work = root / "output" / "pyinstaller"
    work.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--console",
               "--name", "mitsuba-video", "--distpath", str(output), "--workpath", str(work / "build"),
               "--specpath", str(work), "--runtime-hook", str(root / "build_support/runtime.py")]
    command.extend(["--collect-all", "mitsuba_video"])
    # These libraries generate types dynamically. Importing every submodule during collection can initialize
    # every rendering variant; preserve their runtime trees and identify the native entry modules explicitly.
    for name in ("mitsuba", "drjit"):
        distribution = metadata.distribution(name)
        folder = Path(distribution.locate_file(name))
        command.extend(["--add-data", f"{folder}{';' if sys.platform == 'win32' else ':'}{name}"])
        for file in distribution.files or ():
            if str(file).lower().endswith(".dll") and not str(file).replace("\\", "/").startswith(name + "/"):
                command.extend(["--add-binary", f"{distribution.locate_file(file)};bin"])
    for module in ("drjit._drjit_ext", "drjit.scalar", "drjit.llvm", "drjit.cuda", "drjit.cuda.ad",
                   "mitsuba.mitsuba_alias", "mitsuba.mitsuba_ext", "mitsuba.scalar_rgb",
                   "mitsuba.llvm_ad_rgb", "mitsuba.cuda_ad_rgb"):
        command.extend(["--hidden-import", module])
    for name in ("mitsuba-video", "mitsuba", "drjit", "jsonschema"):
        command.extend(["--copy-metadata", name])
    for name in ("pytest", "IPython", "jupyterlab", "notebook", "matplotlib", "tkinter"):
        command.extend(["--exclude-module", name])
    if args.usd:
        command.extend(["--collect-all", "pxr", "--copy-metadata", "usd-core"])
    else:
        command.extend(["--exclude-module", "pxr"])
    if args.ffmpeg:
        ffmpeg = args.ffmpeg.resolve(strict=True)
        command.extend(["--add-binary", f"{ffmpeg}{';' if sys.platform == 'win32' else ':'}bin"])
    command.append(str(root / "build_support/entrypoint.py"))
    subprocess.run(command, cwd=root, check=True)
    application = output / "mitsuba-video"
    shutil.copy2(root / "LICENSE", application / "LICENSE.txt")
    shutil.copy2(root / "docs/install.md", application / "INSTALL.md")
    version = metadata.version("mitsuba-video")
    archive = shutil.make_archive(str(output / f"mitsuba-video-{version}-{sys.platform}"), "zip",
                                  root_dir=output, base_dir="mitsuba-video")
    print(f"Application: {application}")
    print(f"Portable ZIP: {archive}")


if __name__ == "__main__":
    main()
