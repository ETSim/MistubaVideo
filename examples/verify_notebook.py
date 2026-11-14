"""Execute the tutorial in this Python environment and save outputs under output/."""

import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/getting_started.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook>", "exec")
    jupyter = root / "output/jupyter"
    kernel = jupyter / "kernels/mitsuba-video"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "MitsubaVideo", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(jupyter)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(root / "output/jupyter-runtime")
    os.environ["IPYTHONDIR"] = str(root / "output/ipython")

    def cell_start(cell, cell_index):
        print(f"Cell {cell_index + 1}/{len(notebook.cells)}: {cell.cell_type}", flush=True)

    try:
        NotebookClient(notebook, timeout=600, startup_timeout=60, kernel_name="mitsuba-video",
                       resources={"metadata": {"path": str(root)}}, on_cell_start=cell_start).execute()
    finally:
        nbformat.write(notebook, root / "output/getting_started.executed.ipynb")
    print("All notebook cells executed successfully", flush=True)


if __name__ == "__main__":
    main()
