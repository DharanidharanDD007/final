# Real-Time Deepfake Detection Chrome Extension (Manifest V3)

A client-side, zero-server Chrome Extension designed to detect deepfakes and AI face manipulation in real-time during live video conferencing (Google Meet, Microsoft Teams, and Zoom).

---

## Architecture Overview

- **100% Client-Side Privacy**: Runs locally in the browser using WebGL hardware acceleration; no video or audio feeds leave your machine.
- **Manifest V3 Sandboxed Engine**: Employs a sandboxed iframe (`scanner.html` & `scanner.js`) to bypass Chrome's strict Content Security Policy (CSP) while enabling high-performance TensorFlow.js and MediaPipe execution.
- **Artifact-Aware Biometric Tracking**: MediaPipe FaceMesh isolates 468 facial landmarks and dynamically expands bounding boxes by an intentional **15% safety boundary margin** to capture facial blending seams and compositing edges.
- **Multi-Participant Grid Detection**: Concurrently monitors active caller video elements, filtering out mirrored self-views and cycling frames via zero-copy `ImageBitmap` transfers.
- **Acoustic Consistency & Multimodal Fusion**: Uses the Web Audio API (`AudioContext`, `AnalyserNode`) to extract spectral centroids, RMS energy, and high-frequency distributions, combining visual and audio integrity into a unified Trust Score ($w_v=0.75, w_a=0.25$) with automatic silence fallback.
- **In-Call Glassmorphism HUD**: An in-call dashboard (`ui.css`) that displays per-participant trust progress bars, visual/audio chips, and alert tags.

---

## File Structure

```
├── manifest.json         # Chrome Extension Manifest V3 configuration
├── background.js        # Extension background service worker
├── content.js           # DOM video selector, Web Audio capture, HUD injector
├── scanner.html         # Sandboxed execution container
├── scanner.js           # MediaPipe FaceMesh & TensorFlow.js inference engine
├── ui.css               # In-call Glassmorphism HUD stylesheet
├── tf.min.js            # TensorFlow.js v3.18.0 library bundle
├── face_mesh.js         # MediaPipe FaceMesh library bundle
├── train_cnn.py         # Dual-mode batch CNN training & TF.js export pipeline
├── convert_model.py     # H5 to TensorFlow.js conversion script
├── generate_data.py     # Synthetic data generation utility
├── generate_icons.py    # Zero-dependency extension icon generator
├── icons/               # 16x16, 48x48, 128x128 extension security badges
└── models/              # Web-ready TensorFlow.js model weights and graph
    ├── model.json
    └── group1-shard1of1.bin
```

---

## Quick Start

### 1. Load the Extension in Chrome
1. Open Google Chrome and go to `chrome://extensions`.
2. Enable **Developer mode** in the top-right toggle.
3. Click **Load unpacked** and select this directory.
4. Join any meeting on [Google Meet](https://meet.google.com), [Microsoft Teams](https://teams.microsoft.com), or [Zoom](https://zoom.us).

### 2. Model Training (Optional)
To train or recalibrate the CNN model on your custom dataset:
```bash
# Train on a structured dataset (dataset/real and dataset/fake)
python train_cnn.py --data_dir ./dataset --epochs 15 --batch_size 32

# Or run with the built-in synthetic calibration generator
python train_cnn.py --epochs 10
```
This updates `deepfake_cnn.h5` and regenerates the web-optimized weights in `./models/`.
