"""Set library search paths for Mitsuba and an optional bundled FFmpeg."""

import os
import sys
from pathlib import Path

bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
_dll_handles = []
for folder in (bundle, bundle / "mitsuba", bundle / "drjit", bundle / "bin"):
    if folder.is_dir():
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        if sys.platform == "win32":
            _dll_handles.append(os.add_dll_directory(str(folder)))
sys._mitsuba_video_dll_handles = _dll_handles
