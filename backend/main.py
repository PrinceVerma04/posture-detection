"""
Live pose-estimation server.

Two capture modes, one shared RTMPose inference pipeline (via rtmlib,
the official lightweight ONNX Runtime deployment library for
OpenMMLab's mmpose / RTMPose models):

  * "webcam"    - the browser captures frames locally (getUserMedia)
                  and streams them to the server for inference; only
                  keypoints go back, so the browser draws the overlay
                  on top of the live <video> element itself (low
                  latency, no re-encoded video round trip).

  * "ip_webcam" - the server opens the IP camera's MJPEG/RTSP stream
                  directly with OpenCV (browsers can't safely pull raw
                  pixels out of a cross-origin MJPEG <img> due to
                  canvas tainting), runs inference, draws the skeleton
                  on the frame server-side, and streams annotated JPEG
                  frames back over the websocket.
"""
from __future__ import annotations

import os

import asyncio
import base64
import json
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pose_engine import PoseEngine, TrackState

app = FastAPI(title="Live Pose Estimation")

engine = PoseEngine()


# ------------------------------------------------------------------
# Static frontend
# ------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse("../frontend/index.html")


app.mount("/static", StaticFiles(directory="../frontend"), name="static")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def decode_b64_jpeg(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg_b64(frame: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class IPCamWorker:
    """Pulls an IP camera stream and runs inference on a background thread,
    handing annotated JPEG frames back over a thread-safe queue.

    A separate "grabber" thread continuously reads from the stream and keeps
    only the single most recent frame. Without this, OpenCV's internal
    decode buffer for network streams (MJPEG/RTSP) accumulates a backlog
    whenever inference is slower than the camera's frame rate, and the feed
    drifts further and further behind real time the longer it runs. Always
    processing "whatever is newest" keeps latency bounded instead of
    growing.
    """

    def __init__(self, url: str, engine: PoseEngine, kpt_thr: float, draw: bool):
        self.url = url
        self.engine = engine
        self.kpt_thr = kpt_thr
        self.draw = draw
        self.track = engine.new_track_state()
        self.out_q: "queue.Queue[dict]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._new_frame = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _grab_loop(self, cap: cv2.VideoCapture) -> None:
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                self._push({"type": "error", "message": "stream read failed / ended"})
                self._stop.set()
                return
            with self._frame_lock:
                self._latest_frame = frame
            self._new_frame.set()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            self._push({"type": "error", "message": f"could not open stream: {self.url}"})
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass

        grabber = threading.Thread(target=self._grab_loop, args=(cap,), daemon=True)
        grabber.start()

        last_t = time.time()
        fps = 0.0
        while not self._stop.is_set():
            if not self._new_frame.wait(timeout=1.0):
                continue
            self._new_frame.clear()
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                continue

            keypoints, scores = self.engine.infer(frame, track=self.track)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now
            if self.draw:
                frame = self.engine.draw(frame, keypoints, scores, self.kpt_thr)
            try:
                payload = {
                    "type": "annotated_frame",
                    "image": encode_jpeg_b64(frame),
                    "num_people": int(len(keypoints)),
                    "fps": round(fps, 1),
                }
                self._push(payload)
            except Exception as exc:  # noqa: BLE001
                self._push({"type": "error", "message": str(exc)})
        cap.release()

    def _push(self, payload: dict) -> None:
        try:
            self.out_q.put_nowait(payload)
        except queue.Full:
            try:
                self.out_q.get_nowait()
            except queue.Empty:
                pass
            self.out_q.put_nowait(payload)


@app.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket) -> None:
    await websocket.accept()
    ip_worker: Optional[IPCamWorker] = None
    kpt_thr = 0.3
    draw_server_side = True
    track = engine.new_track_state()

    async def drain_ip_worker() -> None:
        nonlocal ip_worker
        assert ip_worker is not None
        loop = asyncio.get_event_loop()
        while ip_worker is not None and not ip_worker._stop.is_set():
            payload = await loop.run_in_executor(None, ip_worker.out_q.get)
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                break

    ip_task: Optional[asyncio.Task] = None
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            mtype = data.get("type")

            if mtype == "config":
                kpt_thr = float(data.get("kpt_thr", kpt_thr))
                if "model" in data:
                    engine.set_mode(data["model"])
                await websocket.send_text(json.dumps({"type": "config_ack", "kpt_thr": kpt_thr, "model": engine.mode}))

            elif mtype == "frame":
                # Browser-captured webcam frame -> run inference, return keypoints only.
                frame = decode_b64_jpeg(data["image"])
                if frame is None:
                    continue
                t0 = time.time()
                keypoints, scores = await asyncio.to_thread(engine.infer, frame, track)
                infer_ms = (time.time() - t0) * 1000
                await websocket.send_text(json.dumps({
                    "type": "keypoints",
                    "keypoints": [kp.tolist() for kp in keypoints],
                    "scores": [s.tolist() for s in scores],
                    "skeleton": engine.skeleton_links,
                    "infer_ms": round(infer_ms, 1),
                }))

            elif mtype == "start_ip":
                if ip_worker is not None:
                    ip_worker.stop()
                    if ip_task:
                        ip_task.cancel()
                url = data["url"]
                draw_server_side = bool(data.get("draw", True))
                ip_worker = IPCamWorker(url, engine, kpt_thr, draw_server_side)
                ip_worker.start()
                ip_task = asyncio.create_task(drain_ip_worker())

            elif mtype == "stop":
                if ip_worker is not None:
                    ip_worker.stop()
                    if ip_task:
                        ip_task.cancel()
                    ip_worker = None
                    ip_task = None

    except WebSocketDisconnect:
        pass
    finally:
        if ip_worker is not None:
            ip_worker.stop()
        if ip_task:
            ip_task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=8000)
