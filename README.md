# Mitsuba Rendering Application

A Python application for rendering 3D scenes with Mitsuba 3, focusing on batch processing of OBJ files and creating animations.

## Features

- Render sequences of OBJ files with Mitsuba 3
- Create videos and GIFs from rendered frames
- Support for both CPU and CUDA GPU rendering
- Configurable rendering parameters (resolution, samples per pixel)
- Multi-threaded rendering for improved performance
- Detailed timing reports and performance statistics
- Field and wear videos of moving bodies from any project (`mitsuba-video`, see [Field and wear videos](#field-and-wear-videos-mitsuba-video))

## Installation

### Prerequisites

- Python 3.8 or higher
- [Mitsuba 3](https://www.mitsuba-renderer.org/) 
- FFmpeg (for video and GIF creation)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/ETSim/MistubaVideo
   cd mitsuba
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Render multiple OBJ files from a folder:

```bash
python main.py multi --folder exports --regex "\d+\.obj"
```

### Command-line Options

```
Options:
  -f, --folder TEXT               Folder containing OBJ files  [required]
  -r, --regex TEXT                Regex pattern for OBJ filenames  [default: \d+\.obj]
  -n, --name TEXT                 Base name for output files  [default: multi_obj]
  -t, --threaded                  Enable multi-threading
  -w, --workers INTEGER           Max worker threads (if threaded)
  --video / --no-video            Enable video creation  [default: video]
  --gif / --no-gif                Enable GIF creation  [default: gif]
  -d, --device TEXT               Rendering device (cpu/cuda)  [default: cpu]
  -s, --spp INTEGER               Samples per pixel  [default: 256]
  --width INTEGER                 Output image width  [default: 1920]
  --height INTEGER                Output image height  [default: 1080]
  -o, --output TEXT               Output directory  [default: output]
  --max-depth INTEGER             Maximum ray bounces  [default: 8]
```

### Additional Commands

Show information about the Mitsuba configuration:

```bash
python main.py info
```

## Field and wear videos (`mitsuba-video`)

[`packages/mitsuba_video`](packages/mitsuba_video/README.md) is a generic renderer for rigid bodies whose UV atlases carry a per-texel field (wear, damage, temperature...). For each frame it:

- path-traces the bodies at their recorded poses, with a field-driven material blend and heatmap;
- adds a UV atlas panel and a covered-area chart.

It then encodes an mp4. Unlike `multi`, it uses textures and per-frame transforms.

Inputs come from pluggable sources:

- **`manifest`:** a versioned JSON any project can write with `ManifestWriter`.
- **`texturefriction`:** TextureFriction `--record` runs.

```bash
pip install -e "packages/mitsuba_video[texturefriction]"     # installs `mitsuba-video` and `tf-render`
mitsuba-video fixture demo && mitsuba-video render demo -q preview
python main.py field render path/to/manifest.json -q paper --camera orbit --title "..."
python main.py wear  render path/to/simulation_<ts> -q paper   # TextureFriction recordings
```

TextureFriction (ETSim/TextureWear) installs this package pinned by commit (`tools/render-requirements.txt` there). On Windows, prefer `mitsuba-video` / `tf-render` over `main.py field|wear` for long renders: they exit cleanly despite Mitsuba's teardown crash.

## Configuration

The application can be configured through a `config.json` file:

```json
{
    "OUTPUT_FOLDER": "output",
    "EXR_FOLDER": "exr",
    "PNG_FOLDER": "png",
    "VIDEO_FOLDER": "video",
    "GIF_FOLDER": "gif",
    "SCENE_FOLDER": "scenes",
    "MESH_FOLDER": "meshes",
    "FRAMERATE": 30,
    "MULTI_THREADED": false,
    "ENABLE_GIF": true,
    "ENABLE_VIDEO": true,
    "ENABLE_EXR": true,
    "ENABLE_PNG": true
}
```

## Project Structure

```
mitsuba/
├── main.py                # Main CLI entry point
├── config.json            # Application configuration
├── src/
│   ├── assets/            # Static assets (scenes, schemas)
│   ├── env/               # Environment map handling
│   ├── processing/        # Processing utilities
│   │   ├── ffmpeg.py      # FFmpeg video processing
│   │   ├── image.py       # Image conversion
│   │   ├── meshes.py      # Mesh file handling
│   │   ├── scene.py       # Scene XML generation
│   │   └── xml.py         # XML processing
│   ├── renderers/         # Renderer implementations
│   │   ├── base.py        # Base renderer class
│   │   └── multi.py       # Multi-object renderer
│   └── utils/             # Utility modules
│       ├── config.py      # Configuration handling
│       ├── constants.py   # Application constants
│       ├── environment.py # Environment setup
│       ├── folder.py      # Folder operations
│       ├── logger.py      # Logging utilities
│       ├── mitsuba.py     # Mitsuba interface
│       ├── timing.py      # Performance timing
│       └── zip.py         # ZIP file handling
└── tests/                 # Unit tests
```

## Docker Support

A Dockerfile and docker-compose.yml are provided for containerized execution:

```bash
# Build the container
docker-compose build

# Run the application
docker-compose run mitsuba python main.py multi --folder /data/exports
```

# Multiple Camera Views in Mitsuba Renderer

This guide explains how to render objects from multiple camera angles using the Mitsuba renderer.

## Available Camera Views

The following camera views are available:

- `front` - View from the front (positive Z axis)
- `back` - View from the back (negative Z axis)
- `left` - View from the left side (negative X axis)
- `right` - View from the right side (positive X axis)
- `top` - View from above (positive Y axis)
- `bottom` - View from below (negative Y axis)
- `perspective` - Default 3/4 perspective view (from a corner)

## Command-line Usage

To render with specific camera views, use the `--view` option (can be specified multiple times):

```bash
python main.py multi --folder models --view front --view top --view right
```

To render all standard camera views in one command:

```bash
python main.py multi --folder models --all-views
```

## Example Workflows

### Orthographic Views for Technical Documentation

To generate orthographic views for technical documentation:

```bash
python main.py multi --folder models --view front --view top --view right --view left
```

### Full 360° Product Visualization

To create a complete 360° visualization:

```bash
python main.py multi --folder models --all-views
```

### Customizing Camera Distance

The default camera distance is 4.0 units. You can adjust this in the code by modifying the `CameraUtils.get_camera_transform` function.

## Programmatic Usage

You can also use the camera view functionality in your Python code:

```python
from src.mitsuba.camera import CameraUtils

# Get a camera transform for a specific view
transform = CameraUtils.get_camera_transform("front", distance=5.0)

# Format the transform for XML
xml_transform = CameraUtils.format_transform_for_xml(transform)
```

## Rendered Output

For each camera view, the renderer will create separate output files with the view name in the filename. For example:

- `multi_obj_frame_0_front.png`
- `multi_obj_frame_0_top.png`
- `multi_obj_frame_0_right.png`


## License

[MIT License](LICENSE)
