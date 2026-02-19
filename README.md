ImageGen GUI & Pipeline

A modular, procedural image processing pipeline capable of generating novel visual styles, enhancing photos, and processing real-time video/screen streams. Built with Python, PyQt5, and FFmpeg.
✨ Features

    Modular Pipeline Architecture: Chain multiple generators together (e.g., photo_enhance -> palette_voronoi -> edge_art).

    GUI (PyQt5):

        Drag-and-drop pipeline ordering.

        Real-time parameter tuning with sliders/inputs.

        Instant visual feedback.

        Save/Load images.

    CLI (Command Line):

        Batch processing.

        Benchmarking tools.

        Live Streaming: Capture screen regions or windows and process them in real-time.

    Advanced Generators:

        Voronoi: Content-aware tessellation and pointillism effects.

        Content Context: Style transfer using learned color profiles.

        Filters: Warp, flow, and edge detection.

        Learning: Extract color palettes and composition data from images.

    Video Support: Full FFmpeg integration for reading/writing video files and capturing windows natively.

🛠️ Installation
1. Clone the Repository

Important: This repository uses Git LFS to store FFmpeg binaries. Ensure you have Git LFS installed before cloning.
Bash

git lfs install
git clone https://github.com/YourUsername/ImageGenGUI.git
cd ImageGenGUI

2. Set up Python Environment

It is recommended to use a virtual environment.
Bash

# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Activate venv (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

3. Verify FFmpeg

For the application to work (especially video and building the EXE), the ffmpeg-8.0-essentials_build folder must be present in the root directory.

    If you cloned with LFS, this should already be there.

    If not, download an FFmpeg static build and rename the folder to exactly ffmpeg-8.0-essentials_build.

🚀 Usage
Running the GUI

The GUI is the easiest way to experiment with pipelines.
Bash

python gui.py

    Left Panel: Add/Remove/Reorder processing stages.

    Center: View the result.

    Right Panel: Tweak parameters for the selected stage.

Running the CLI

Use main.py for headless operations, batch processing, or streaming.

Process a single image:
Bash

python main.py run --url "input.jpg" --pipeline "photo_enhance|palette_voronoi" --out "output.png"

Stream screen capture (Windows):
Bash

python main.py stream --pipeline "edge_art" --preview --fps 30

Capture specific window:
Bash

python main.py stream --window "Notepad" --pipeline "mosaic" --preview

📦 Building Standalone EXE

You can bundle the application, dependencies, and FFmpeg into a single .exe file using PyInstaller.

    Ensure you have the ffmpeg-8.0-essentials_build folder in the project root.

    Run the build command:

Bash

pyinstaller imagegen.spec

    The executable will be generated in the dist/ folder:

        dist/ImageGenGUI.exe

Note: The spec file is configured to hide the console window and bundle FFmpeg internally.
🧩 Generator Modules
Module	Description
palettes.py	Core Voronoi, Mosaic, and FBM noise generators.
content.py	Context-aware styling and photo enhancement/upscaling logic.
filters.py	Distortion effects, warping, and flow fields.
learning.py	Profile extraction (analyzes an image to learn its palette/composition).
📋 Requirements

    Python 3.8+

    PyQt5

    Pillow (PIL)

    NumPy

    Requests

    OpenCV (optional, for preview window)

    MSS (optional, for screen capture)
