"""
build_desktop_app.py – Builds the standalone Windows desktop app.

Usage:
  pip install -r requirements.txt
  pip install -r requirements-desktop.txt
  python download_bundled_models.py   (only needed once, or if bundled_models/ is missing)
  python build_desktop_app.py

Output: dist/ReferenceValidator/ — zip this whole folder to distribute it.
ReferenceValidator.exe inside it is the entry point; it depends on
everything else in that folder (_internal/), so the folder must be shared
as a whole, not just the .exe by itself.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED_MODELS = os.path.join(HERE, "bundled_models")

if not os.path.isdir(BUNDLED_MODELS):
    print("bundled_models/ not found — fetching model files first...")
    subprocess.check_call([sys.executable, os.path.join(HERE, "download_bundled_models.py")])

cmd = [
    sys.executable, "-m", "PyInstaller", "desktop_app.py",
    "--name", "ReferenceValidator",
    "--onedir",
    "--noconfirm",
    "--add-data", "reference_validator-GUIDE-1.py;.",
    "--add-data", "app_ui.html;.",
    "--add-data", "bundled_models;bundled_models",
    "--collect-all", "onnxruntime",
    "--collect-all", "optimum",
    "--collect-all", "transformers",
    "--collect-all", "tokenizers",
    "--collect-all", "sentence_transformers",
    "--collect-all", "huggingface_hub",
    "--collect-all", "safetensors",
    "--hidden-import", "flask_cors",
    "--hidden-import", "webview",
    # Add "--windowed" here for a final release build to hide the console
    # window — leave it out while testing so errors are visible.
]

print("Running:", " ".join(cmd))
subprocess.check_call(cmd, cwd=HERE)
print("\nBuild complete. See dist/ReferenceValidator/")
