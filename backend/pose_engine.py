"""Thin wrapper around rtmlib's RTMDet+RTMPose ('Body') pipeline.

rtmlib is the official lightweight ONNX Runtime deployment library
published by the OpenMMLab / mmpose team, running the same RTMPose
models mmpose trains, without requiring torch/mmcv at inference time.

Person detection (RTMDet/YOLOX) is far more expensive than pose
keypoint estimation (~130ms vs ~40ms on CPU for the lightweight
model), so for live video we only re-run the detector every
`det_interval` frames and reuse (padded) boxes from the last detection
in between -- this is the standard tracking-by-detection-interval
trick and cuts steady-state per-frame latency roughly 4x.
"""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np
from rtmlib import Body, draw_skeleton

# COCO-17 skeleton connectivity used by rtmlib's Body() model.
COCO17_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

VALID_MODES = {"lightweight", "balanced", "performance"}


class TrackState:
    """Per-stream detection cache (one instance per websocket / camera)."""

    __slots__ = ("frame_count", "last_bboxes")

    def __init__(self) -> None:
        self.frame_count = 0
        self.last_bboxes: Optional[np.ndarray] = None


def _pad_bboxes(bboxes: np.ndarray, shape: Tuple[int, int], factor: float = 0.2) -> np.ndarray:
    """Widen cached boxes a bit so a few frames of motion don't fall outside
    the crop before the next real detection runs."""
    if bboxes is None or len(bboxes) == 0:
        return bboxes
    h, w = shape[:2]
    out = bboxes.copy()
    bw = out[:, 2] - out[:, 0]
    bh = out[:, 3] - out[:, 1]
    out[:, 0] = np.clip(out[:, 0] - bw * factor / 2, 0, w)
    out[:, 1] = np.clip(out[:, 1] - bh * factor / 2, 0, h)
    out[:, 2] = np.clip(out[:, 2] + bw * factor / 2, 0, w)
    out[:, 3] = np.clip(out[:, 3] + bh * factor / 2, 0, h)
    return out


class PoseEngine:
    def __init__(self, mode: str = "lightweight", device: str = "cpu", det_interval: int = 5):
        self.mode = mode if mode in VALID_MODES else "balanced"
        self.device = device
        self.det_interval = det_interval
        self.skeleton_links = COCO17_SKELETON
        self._lock = threading.Lock()
        self._body = self._build(self.mode)

    def _build(self, mode: str) -> Body:
        return Body(
            mode=mode,
            to_openpose=False,
            backend="onnxruntime",
            device=self.device,
        )

    def set_mode(self, mode: str) -> None:
        if mode not in VALID_MODES or mode == self.mode:
            return
        with self._lock:
            self._body = self._build(mode)
            self.mode = mode

    def new_track_state(self) -> TrackState:
        return TrackState()

    def infer(self, frame: np.ndarray, track: Optional[TrackState] = None):
        """Run detection+pose. If `track` is given, the detector only runs
        every `det_interval` frames (or if no box was found last time) and
        cached boxes are reused (padded) in between for lower latency."""
        with self._lock:
            body = self._body
            if track is None:
                bboxes = body.det_model(frame)
                keypoints, scores = body.pose_model(frame, bboxes=bboxes)
                return keypoints, scores

            need_detect = (
                track.last_bboxes is None
                or len(track.last_bboxes) == 0
                or track.frame_count % self.det_interval == 0
            )
            if need_detect:
                bboxes = body.det_model(frame)
                track.last_bboxes = bboxes
            else:
                bboxes = _pad_bboxes(track.last_bboxes, frame.shape)
            track.frame_count += 1
            keypoints, scores = body.pose_model(frame, bboxes=bboxes)
        return keypoints, scores

    def draw(self, frame: np.ndarray, keypoints, scores, kpt_thr: float = 0.3) -> np.ndarray:
        return draw_skeleton(frame, keypoints, scores, kpt_thr=kpt_thr, openpose_skeleton=False)
