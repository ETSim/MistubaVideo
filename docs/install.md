# Install MitsubaVideo

## Python package

From a checkout, install the general renderer with Python 3.10 or newer:

```bash
python -m pip install .
mitsuba-video doctor
mitsuba-video fixture output/cube --frames 24
mitsuba-video render output/cube -q preview --no-encode
```

For development, use `python -m pip install -e ".[test]"`. Optional extras are `usd`, `notebook` and `freeze`.
The older `packages/mitsuba_video` installation path remains supported for downstream repositories.

You can also install a built wheel with `python -m pip install path/to/mitsuba_video-0.4.0-py3-none-any.whl`.
The publishing workflow builds the wheel and source archive. A PyPI release requires trusted-publisher setup;
until one exists, use the wheel or install this repository directly:

```bash
python -m pip install "git+https://github.com/ETSim/MistubaVideo.git@codex/general-renderer-plugins"
```

## Windows executable

Extract the portable ZIP and keep the complete `mitsuba-video` folder together. Run `mitsuba-video.exe` in a
terminal; Python is included in the bundle, so the target computer does not need a separate Python installation.
The `_internal` directory holds native renderer libraries and resources and must stay beside the executable.

```powershell
.\mitsuba-video.exe doctor
.\mitsuba-video.exe tui
.\mitsuba-video.exe fixture output\cube --frames 24
.\mitsuba-video.exe render output\cube -q preview --no-encode
```

CPU scalar rendering is available without a GPU. CUDA needs a compatible NVIDIA driver. LLVM CPU acceleration
needs a usable system LLVM runtime. The executable cannot install arbitrary Python source plugins; use the pip
installation for that workflow. USD is included only in builds made with `--usd`.

FFmpeg is needed to encode MP4 and GIF files. Install it on PATH, or build with `--ffmpeg PATH` to include a copy.
Building with an external FFmpeg binary also requires complying with that binary's redistribution license.

## Build distributions

```bash
python -m pip install -e ".[freeze]" build
python -m build
python examples/build_executable.py
```

The wheel and source archive go to `dist/`. On Windows, the executable and its libraries go to
`dist/windows/mitsuba-video/`, and the portable ZIP goes to `dist/windows/`.

For an executable with USD, install `.[usd,freeze]` and pass `--usd`. Build on the target operating system;
the Windows workflow creates Windows executables and does not require Python on the destination machine.
These executables are unsigned, so Windows may show an unknown-publisher prompt.
