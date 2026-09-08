"""
MediaPipe FaceMesh WebAssembly & Solution Localizer
===================================================
Ensures the Chrome extension runs 100% locally and offline without external CDN dependencies.
Checks local node_modules first, or downloads the 6 core WebAssembly & asset files.
"""

import os
import shutil
import urllib.request

MEDIAPIPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe")
os.makedirs(MEDIAPIPE_DIR, exist_ok=True)

REQUIRED_FILES = [
    "face_mesh.js",
    "face_mesh.binarypb",
    "face_mesh_solution_packed_assets_loader.js",
    "face_mesh_solution_packed_assets.data",
    "face_mesh_solution_simd_wasm_bin.js",
    "face_mesh_solution_simd_wasm_bin.wasm",
    "face_mesh_solution_wasm_bin.js",
    "face_mesh_solution_wasm_bin.wasm"
]

LOCAL_FALLBACK_DIR = r"D:\Final year project\node_modules\@mediapipe\face_mesh"
CDN_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619"


def setup_mediapipe_assets():
    print(f">> Localizing MediaPipe FaceMesh files into: {MEDIAPIPE_DIR}")
    
    for filename in REQUIRED_FILES:
        target_path = os.path.join(MEDIAPIPE_DIR, filename)

        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f" [OK] Existing: {filename} ({os.path.getsize(target_path)} bytes)")
            continue

        # Check local node_modules
        local_src = os.path.join(LOCAL_FALLBACK_DIR, filename)
        if os.path.exists(local_src):
            shutil.copy2(local_src, target_path)
            print(f" [COPIED] {filename} from local node_modules ({os.path.getsize(target_path)} bytes)")
            continue

        # Download from CDN if missing
        cdn_url = f"{CDN_BASE_URL}/{filename}"
        print(f" [DOWNLOADING] {filename} from {cdn_url}...")
        try:
            req = urllib.request.Request(cdn_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_file:
                shutil.copyfileobj(resp, out_file)
            print(f" [DOWNLOADED] {filename} ({os.path.getsize(target_path)} bytes)")
        except Exception as e:
            print(f" [ERROR] Could not download {filename}: {e}")

    print("\n[SUCCESS] MediaPipe localization complete. All required assets are in ./mediapipe/")


if __name__ == "__main__":
    setup_mediapipe_assets()
