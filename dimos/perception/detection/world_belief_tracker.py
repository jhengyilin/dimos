# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detector-agnostic short-term trackers — the pluggable `Tracker` seam (design §2).

The world-belief consumes a stable per-object `track_id`; WHERE it comes from is a
swappable backend. These run on plain 2D boxes from ANY detector (here: cached
Grounding-DINO boxes — explicitly NOT YOLO's built-in tracker), proving the fix for
the sub-gate look-alike merge is the ABSTRACTION, not a single model's feature.

Backends (all satisfy `update(boxes, scores) -> list[int] track_id`):
  * NoTracker  — the legal no-op (track_id=-1); reproduces the merge baseline.
  * IouTracker — greedy/Hungarian IoU association across frames (ByteTrack-lite).
  * SortTracker — classic SORT: per-object Kalman (constant-velocity) + Hungarian
    IoU assignment (filterpy + scipy). Motion model survives brief missed detections.

Key idea (why this separates what the static position gate can't): each box is matched
to its OWN previous box over ~1/15 s, where inter-frame motion (<1 cm) is far smaller
than the gap between two adjacent look-alikes — so the two cans keep distinct ids.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4); boxes are x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x11, y11, x12, y12 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    x21, y21, x22, y22 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    iw = np.clip(np.minimum(x12, x22) - np.maximum(x11, x21), 0, None)
    ih = np.clip(np.minimum(y12, y22) - np.maximum(y11, y21), 0, None)
    inter = iw * ih
    area_a = np.clip(x12 - x11, 0, None) * np.clip(y12 - y11, 0, None)
    area_b = np.clip(x22 - x21, 0, None) * np.clip(y22 - y21, 0, None)
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


class NoTracker:
    name = "none"

    def update(self, boxes, scores=None):
        return [-1] * len(boxes)


class IouTracker:
    name = "iou"

    def __init__(self, iou_threshold: float = 0.2, max_age: int = 12):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, dict] = {}   # tid -> {box, age, hits}
        self._next = 0

    def update(self, boxes, scores=None):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for t in self.tracks.values():
            t["age"] += 1
        assigned = [-1] * len(boxes)
        tids = list(self.tracks.keys())
        if tids and len(boxes):
            tboxes = np.array([self.tracks[t]["box"] for t in tids], dtype=np.float32)
            iou = _iou_matrix(boxes, tboxes)              # (N_det, N_trk)
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    tid = tids[c]
                    assigned[r] = tid
                    self.tracks[tid].update(box=boxes[r], age=0)
                    self.tracks[tid]["hits"] += 1
        for i in range(len(boxes)):
            if assigned[i] == -1:
                tid = self._next
                self._next += 1
                self.tracks[tid] = {"box": boxes[i], "age": 0, "hits": 1}
                assigned[i] = tid
        for tid in [t for t, v in self.tracks.items() if v["age"] > self.max_age]:
            del self.tracks[tid]
        return assigned


# --------------------------------------------------------------------------- SORT

def _to_z(box):
    """x1,y1,x2,y2 -> [cx,cy,s,r] (center, scale=area, aspect)."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx, cy = box[0] + w / 2.0, box[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float32)


def _to_box(x):
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(max(s, 1e-6) * max(r, 1e-6))
    h = max(s, 1e-6) / (w + 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


class _KalmanBox:
    """Constant-velocity Kalman on [cx,cy,s,r, vx,vy,vs] — canonical SORT model."""

    def __init__(self, box, tid):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.kf.F[i, i + 4] = 1.0
        self.kf.H = np.zeros((4, 7), dtype=np.float32)
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _to_z(box).reshape(4, 1)
        self.tid = tid
        self.age = 0
        self.hits = 1

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        return _to_box(self.kf.x[:4].reshape(-1))

    def update(self, box):
        self.age = 0
        self.hits += 1
        self.kf.update(_to_z(box).reshape(4, 1))

    @property
    def box(self):
        return _to_box(self.kf.x[:4].reshape(-1))


class SortTracker:
    name = "sort"

    def __init__(self, iou_threshold: float = 0.2, max_age: int = 12, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: list[_KalmanBox] = []
        self._next = 0

    def update(self, boxes, scores=None):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        preds = np.array([t.predict() for t in self.trackers], dtype=np.float32).reshape(-1, 4)
        assigned = [-1] * len(boxes)
        if len(self.trackers) and len(boxes):
            iou = _iou_matrix(boxes, preds)
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    self.trackers[c].update(boxes[r])
                    assigned[r] = self.trackers[c].tid
        for i in range(len(boxes)):
            if assigned[i] == -1:
                trk = _KalmanBox(boxes[i], self._next)
                self._next += 1
                self.trackers.append(trk)
                assigned[i] = trk.tid
        self.trackers = [t for t in self.trackers if t.age <= self.max_age]
        return assigned


def make_tracker(name: str, **kw):
    if name == "none":
        return NoTracker()
    return {"iou": IouTracker, "sort": SortTracker}[name](**kw)


def assign_track_ids(tracker, detections) -> None:
    """Stamp track_id onto a list of dimos 2D/3D detections IN PLACE, from ANY detector.

    Each detection must expose `.bbox` as (x1, y1, x2, y2) and a writable `.track_id`.
    This is the detector-agnostic alternative to a model's built-in tracker: run the detector
    in plain predict() mode (no internal tracking) and let this seam own identity continuity,
    which WorldBelief's priority-1 track_id path then consumes.
    """
    boxes = [tuple(d.bbox) for d in detections]
    tids = tracker.update(boxes)
    for d, t in zip(detections, tids):
        d.track_id = int(t)


__all__ = ["NoTracker", "IouTracker", "SortTracker", "make_tracker", "assign_track_ids"]
