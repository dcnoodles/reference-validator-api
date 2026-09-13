"""
download_bundled_models.py – Fetches the two cross-encoder models' ONNX
files into bundled_models/, for the desktop app build to embed directly.
Run this once before building (build_desktop_app.py handles it automatically).

Only pulls the files actually needed at runtime (config/tokenizer files +
the quantized int8 ONNX weights) — skips the full-precision PyTorch/ONNX
weights, which are never used since the app always loads via the ONNX
backend. Total size: ~166MB for both models.
"""
import os
from huggingface_hub import hf_hub_download

MODELS = {
    "similarity": "cross-encoder/stsb-distilroberta-base",
    "nli":        "cross-encoder/nli-MiniLM2-L6-H768",
}
FILES = [
    "config.json", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "onnx/model_qint8_avx512.onnx",
]

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundled_models")

for local_name, repo_id in MODELS.items():
    dest = os.path.join(BASE_DIR, local_name)
    os.makedirs(os.path.join(dest, "onnx"), exist_ok=True)
    print(f"Fetching {repo_id} -> {dest}")
    for fname in FILES:
        path = hf_hub_download(repo_id=repo_id, filename=fname)
        out_path = os.path.join(dest, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(path, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        print(f"  {fname}")

print("\nDone. bundled_models/ is ready for the desktop app build.")
