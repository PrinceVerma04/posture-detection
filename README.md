# Live Pose Estimation

A complete, self-hosted, browser-based live human pose estimation app,
with support for both a built-in/USB webcam and any IP webcam (e.g.
the Android "IP Webcam" app, or RTSP cameras).

Pose inference is powered by **RTMPose** (RTMDet person detector +
RTMPose keypoint model), the same models trained and published by the
[OpenMMLab / mmpose](https://github.com/open-mmlab/mmpose) project,
run here via [`rtmlib`](https://github.com/Tau-J/rtmlib) — mmpose's
official lightweight ONNX Runtime deployment library. This avoids
needing a full `torch` + `mmcv` + `mmdet` + `mmpose` install (which is
fragile to compile against very new GPUs/Python versions) while using
the exact same models.

## Architecture

```
┌─────────────┐   getUserMedia frames    ┌──────────────────┐
│   Browser   │ ───────────────────────► │  FastAPI backend │
│  (webcam)   │ ◄─────────────────────── │  (WebSocket)     │
└─────────────┘   keypoints only         │                  │
                                          │  RTMDet + RTMPose│
┌─────────────┐   "connect to this URL"  │  (rtmlib, ONNX)  │
│   Browser   │ ───────────────────────► │                  │
│ (IP webcam) │ ◄─────────────────────── └──────────────────┘
└─────────────┘   annotated JPEG frames        │
                                                ▼
                                     server pulls the IP camera's
                                     MJPEG/RTSP stream directly
                                     with OpenCV, runs inference,
                                     draws the skeleton, streams
                                     annotated frames back
```

* **Built-in/USB webcam**: captured client-side (`getUserMedia`), sent
  to the server as JPEG frames over a WebSocket for inference. Only
  keypoints (small JSON) come back, and the browser draws the
  skeleton overlay on a `<canvas>` on top of the live `<video>` —
  keeps bandwidth and latency low.
* **IP webcam**: the browser cannot safely read raw pixels out of a
  cross-origin MJPEG `<img>` (canvas tainting / CORS), so instead the
  server opens the stream directly with OpenCV (`cv2.VideoCapture`),
  runs inference and draws the overlay itself, and streams annotated
  JPEG frames back over the same WebSocket for the browser to display.

## Setup

Everything is already set up in this directory:

- `.venv/` — Python 3.10 virtualenv (chosen for best wheel compatibility;
  Python 3.14, the system default, doesn't yet have solid wheels for this
  stack) with `fastapi`, `uvicorn`, `opencv-python-headless`, `rtmlib`,
  `onnxruntime-gpu`, and the CUDA runtime pip packages installed.
- `backend/` — FastAPI app (`main.py`) + pose inference wrapper (`pose_engine.py`).
- `frontend/` — single-page UI (`index.html`), no build step.

To reinstall from scratch:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Running

```bash
./run.sh
```

Then open **http://localhost:8000** in a browser (Chrome/Firefox/Edge).
If accessing from another device on your network, use the host
machine's LAN IP instead of `localhost`, and grant camera permission
over HTTPS or `localhost` only — browsers block `getUserMedia` on
plain HTTP for non-localhost origins.

## Using it

1. **Built-in/USB camera**: select "Built-in / USB", pick a device
   (populated after you grant camera permission), click **Start
   Webcam**.
2. **IP webcam**: select "IP Webcam", enter the stream URL (e.g.
   `http://192.168.1.23:8080/video` for the Android IP Webcam app's
   MJPEG endpoint, or an RTSP URL), click **Connect**. This is
   processed entirely server-side.
3. Adjust the **RTMPose size** (lightweight/balanced/performance) and
   **keypoint confidence** threshold live from the left panel.

## Notes / known limitations

- **GPU acceleration**: `onnxruntime-gpu` and the CUDA 12 runtime pip
  packages are installed, and `LD_LIBRARY_PATH` is wired up in
  `run.sh`, but in this environment (RTX 5080 / very new driver stack)
  the ONNX Runtime CUDA execution provider did not register at
  runtime, so inference currently runs on **CPU**. This is still fast
  enough for a live demo with the "lightweight" model (~10-15 fps).
  To debug further: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
  should list `CUDAExecutionProvider`; if it doesn't, try matching
  `onnxruntime-gpu`'s exact required CUDA/cuDNN minor version, or
  build ONNX Runtime from source against this driver.
- First run downloads the RTMDet + RTMPose ONNX model weights from
  OpenMMLab's storage (cached under `~/.cache/rtmlib/hub/checkpoints/`
  after that).
- The pose model is single-view, 2D, COCO-17 keypoints (body only, no
  hands/face). Swapping to mmpose's `Wholebody` model (133 keypoints)
  is a small change in `backend/pose_engine.py` if needed later.
