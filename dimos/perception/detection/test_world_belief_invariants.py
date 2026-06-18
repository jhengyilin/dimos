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

"""Identity-integrity REGRESSION on real recorded detections (auto-runs with the suite).

These assert the invariants that must hold for any correct world-belief, replaying the engine over
cached REAL detections. The MERGE invariant is the permanent regression for the hand-wave bug
(priority-1 track_id bypassing co-occurrence -> one id smeared across multiple objects): it was 1827
merge frames with the bug, 0 after the fix. If that class of bug ever returns, this test fails —
no manual video review required.

Skips cleanly if the detection caches are absent (so it never blocks CI without the clips).
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.type.detection3d.object import Object
from dimos.perception.detection.world_belief import WorldBelief
from dimos.perception.detection.world_belief_tracker import make_tracker

CACHE_DIR = "/home/dimos/Documents/0616_object_scene_registraion_modul_2_arch_exp/gate4/cache"
CLIPS = ["clip_session1", "clip_session2", "followup1", "followup2", "endurance1"]
_IMG = Image(np.zeros((4, 4, 3), np.uint8))


def _has(clip: str) -> bool:
    return os.path.exists(f"{CACHE_DIR}/{clip}_grounding_dino.pkl")


def _mk(pos, name, score, ts, tid):
    v = Vector3(pos[0], pos[1], pos[2])
    pc = PointCloud2(pointcloud=o3d.geometry.PointCloud(), frame_id="world", ts=ts)
    return Object(center=v, size=Vector3(0.05, 0.05, 0.1), pose=PoseStamped(position=v), pointcloud=pc,
                  image=_IMG, bbox=(0, 0, 10, 10), track_id=int(tid), class_id=0,
                  confidence=float(score), name=name, ts=float(ts))


def _replay(clip: str):
    cache = pickle.load(open(f"{CACHE_DIR}/{clip}_grounding_dino.pkl", "rb"))
    b = WorldBelief(distance_threshold=0.12, min_support=3, recent_window=1.5,
                    eviction_ttl_s=60.0, cooccurrence_gate=True, enable_history=False)
    tr = make_tracker("iou")
    merge_frames, present_series = 0, []
    for fr in cache["frames"]:
        dets = [d for d in fr["dets"] if d["pos"] is not None]
        tids = tr.update([tuple(d["bbox"]) for d in dets])
        res = b.add_objects([_mk(d["pos"], d["label"], d["score"], fr["ts"], t) for d, t in zip(dets, tids)])
        present_series.append(len(b.get_objects()))
        byid = defaultdict(list)
        for d, o in zip(dets, res):
            byid[o.object_id].append(d["pos"])
        for ps in byid.values():
            if len(ps) >= 2 and any(np.linalg.norm(np.array(ps[i]) - np.array(ps[j])) > 0.04
                                    for i in range(len(ps)) for j in range(i + 1, len(ps))):
                merge_frames += 1
                break
    return merge_frames, present_series


@pytest.mark.parametrize("clip", CLIPS)
def test_inv_no_merge(clip: str) -> None:
    """INV-MERGE: no object_id is ever assigned to >=2 spatially-separated detections in a frame."""
    if not _has(clip):
        pytest.skip(f"cache for {clip} not present")
    merge_frames, _ = _replay(clip)
    assert merge_frames == 0, f"{clip}: {merge_frames} merge frames (one id on >=2 separated objects)"


@pytest.mark.parametrize("clip", CLIPS)
def test_inv_ghost_bounded(clip: str) -> None:
    """INV-GHOST: present-set must shrink sometimes (no monotone ghost accumulation)."""
    if not _has(clip):
        pytest.skip(f"cache for {clip} not present")
    _, present = _replay(clip)
    decreases = sum(1 for i in range(1, len(present)) if present[i] < present[i - 1])
    assert decreases > 0, f"{clip}: present-set never shrank (monotone-growing = ghost pileup)"
