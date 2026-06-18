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

"""Tests for WorldBelief (Arch D engine) — behaviors validated on real data, plus ObjectDB API parity.

Uses REAL dimos ``Object`` instances. Time is driven by ``Object.ts`` (the engine reads obs time), so
these are deterministic and need no wall-clock control.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.objectDB import ObjectDB
from dimos.perception.detection.type.detection3d.object import Object
from dimos.perception.detection.world_belief import WorldBelief, make_belief_engine

_IMG = Image(np.zeros((4, 4, 3), np.uint8))


def _mk(x: float, y: float, z: float, name: str = "can", track_id: int = -1, ts: float = 0.0) -> Object:
    pc = PointCloud2(pointcloud=o3d.geometry.PointCloud(), frame_id="world", ts=ts)
    return Object(
        center=Vector3(x, y, z), size=Vector3(0.05, 0.05, 0.1),
        pose=PoseStamped(position=Vector3(x, y, z)), pointcloud=pc, image=_IMG,
        bbox=(0, 0, 10, 10), track_id=track_id, class_id=0, confidence=0.9, name=name, ts=ts,
    )


def test_support_gate_suppresses_single_detection() -> None:
    """A scattered single detection never enters the present-set (the ghost fix)."""
    b = WorldBelief(distance_threshold=0.12, min_support=3, recent_window=1.0, enable_history=False)
    b.add_objects([_mk(5, 5, 5, track_id=9, ts=0.0)])
    assert len(b.get_objects()) == 0, "single transient detection should not be present"
    for i in range(4):
        b.add_objects([_mk(0, 0, 1, track_id=1, ts=i * 0.1)])
    assert len(b.get_objects()) == 1, "4 sightings within the window should be present"


def test_ghost_is_evicted_after_ttl() -> None:
    """An object that stops being seen is removed from the maintained table after eviction_ttl_s."""
    b = WorldBelief(distance_threshold=0.12, min_support=2, recent_window=1.0,
                    eviction_ttl_s=5.0, track_id_ttl_s=1.0, enable_history=False)
    for i in range(3):
        b.add_objects([_mk(0, 0, 1, track_id=1, ts=i * 0.1)])
    assert len(b.get_all_objects()) == 1
    # A different object far away, 100s later -> advances 'now' -> the old one is evicted.
    b.add_objects([_mk(2, 2, 2, track_id=2, ts=100.0)])
    ids = {o.object_id for o in b.get_all_objects()}
    assert len(ids) == 1, "stale object should be evicted; only the new one remains"


def test_cooccurrence_keeps_lookalikes_apart() -> None:
    """Two same-name objects within the gate but with different track_ids stay TWO ids."""
    def run(cooc: bool, tid_a: int, tid_b: int) -> int:
        b = WorldBelief(distance_threshold=0.12, min_support=2, recent_window=2.0,
                        cooccurrence_gate=cooc, enable_history=False)
        for i in range(4):
            t = i * 0.1
            b.add_objects([_mk(0.0, 0, 1, "can", tid_a, t), _mk(0.06, 0, 1, "can", tid_b, t)])
        return len(b.get_objects())

    assert run(cooc=True, tid_a=1, tid_b=2) == 2, "distinct tracks must not merge under the gate"
    assert run(cooc=True, tid_a=-1, tid_b=-1) == 1, "no track_id -> position merges (baseline)"
    assert run(cooc=False, tid_a=1, tid_b=2) == 1, "co-occurrence off -> merges like baseline"


def test_sticky_identity_single_id_for_moving_object() -> None:
    """One object slid slowly across the table keeps a single object_id (low churn)."""
    b = WorldBelief(distance_threshold=0.12, min_support=2, recent_window=2.0, pos_ema=0.5,
                    enable_history=False)
    ids = set()
    for i in range(8):
        res = b.add_objects([_mk(0.02 * i, 0, 1, "can", track_id=7, ts=i * 0.1)])
        ids |= {o.object_id for o in res}
    assert len(ids) == 1, f"a single slid object should keep one id, got {len(ids)}"


def test_api_parity_with_objectdb() -> None:
    """WorldBelief implements ObjectDB's full public surface and returns real Objects."""
    public = ["add_objects", "get_objects", "get_all_objects", "promote", "find_by_name",
              "find_by_object_id", "find_nearest", "clear", "get_stats", "get_last_add_stats",
              "agent_encode"]
    wb = WorldBelief(enable_history=False)
    for m in public:
        assert hasattr(wb, m), f"WorldBelief missing ObjectDB method {m}"
        assert hasattr(ObjectDB, m), f"baseline ObjectDB missing {m}"
    for i in range(4):
        wb.add_objects([_mk(0, 0, 1, "mug", track_id=3, ts=i * 0.1)])
    objs = wb.get_objects()
    assert objs and all(isinstance(o, Object) for o in objs)
    assert wb.find_by_name("mug") and wb.find_by_object_id(objs[0].object_id) is objs[0]
    assert wb.find_nearest(Vector3(0, 0, 1), "mug") is objs[0]


def test_history_when_entered() -> None:
    """when_entered returns the first-seen time, and still answers after the object leaves the table."""
    b = WorldBelief(distance_threshold=0.12, min_support=2, recent_window=1.0,
                    eviction_ttl_s=3.0, track_id_ttl_s=1.0, enable_history=True)
    res = b.add_objects([_mk(0, 0, 1, "can", track_id=1, ts=10.0)])
    oid = res[0].object_id
    for i in range(1, 4):
        b.add_objects([_mk(0, 0, 1, "can", track_id=1, ts=10.0 + i * 0.1)])
    assert abs(b.when_entered(oid) - 10.0) < 1e-6, "entered_t should be first sighting (10.0)"
    # object leaves and is evicted (advance time far with a different object)
    b.add_objects([_mk(5, 5, 5, "mug", track_id=2, ts=100.0)])
    assert b.find_by_object_id(oid) is None, "old object evicted from table"
    assert abs(b.when_entered(oid) - 10.0) < 1e-6, "history still answers when_entered after eviction"
    b.close()


def test_rehydrate_from_disk_cross_process(tmp_path) -> None:
    """A FRESH engine instance restores identities + entered_t from the on-disk store (cross-process)."""
    db = str(tmp_path / "wb_history.db")
    # session 1: engine A writes history to disk
    a = WorldBelief(distance_threshold=0.12, min_support=2, recent_window=1.0,
                    enable_history=True, history_path=db)
    res = a.add_objects([_mk(0, 0, 1, "can", track_id=1, ts=5.0),
                         _mk(0.5, 0, 1, "mug", track_id=2, ts=5.0)])
    can_id = res[0].object_id
    for i in range(1, 4):
        a.add_objects([_mk(0, 0, 1, "can", track_id=1, ts=5.0 + i * 0.1),
                       _mk(0.5, 0, 1, "mug", track_id=2, ts=5.0 + i * 0.1)])
    assert abs(a.when_entered(can_id) - 5.0) < 1e-6
    a.close()  # simulate process exit; store persists on disk

    # session 2: a BRAND-NEW engine restores from the same on-disk store (no shared RAM)
    b = WorldBelief(distance_threshold=0.12, min_support=2, recent_window=1.0,
                    enable_history=True, history_path=db)
    b.rehydrate()
    restored = {o.object_id for o in b.get_all_objects()}
    assert can_id in restored, "the can identity must be restored from disk by its original object_id"
    assert abs(b.when_entered(can_id) - 5.0) < 1e-6, "entered_t recovered from disk (genuine, not a fresh mint)"
    # the can returns near its prior spot -> re-associates to the SAME restored id, keeps the s1 entry time
    r = b.add_objects([_mk(0, 0, 1, "can", track_id=9, ts=50.0)])  # new track id (tracker also restarted)
    assert r[0].object_id == can_id, "returning object re-associates to the restored identity"
    assert abs(b.when_entered(can_id) - 5.0) < 1e-6, "cross-session entry time preserved after re-sighting"
    b.close()


def test_factory_selects_engine() -> None:
    assert type(make_belief_engine("world_belief")).__name__ == "WorldBelief"
    assert type(make_belief_engine("objectdb")).__name__ == "ObjectDB"
    # factory passes ObjectDB-compat kwargs through to either engine without error
    make_belief_engine("world_belief", distance_threshold=0.1, min_detections_for_permanent=5)
    make_belief_engine("objectdb", distance_threshold=0.1, min_detections_for_permanent=5)
