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

"""WorldBelief — the Arch D object-lifecycle engine, validated on real data.

A drop-in alternative to ``ObjectDB`` (same public API, same ``Object`` type) implementing the
synthesis world-belief design (see the 0616 experiment / GENERAL_SCENE_MODULE_DESIGN.md):

  * MAINTAINED identity assigned ONCE (sticky, support-prioritised association) — low id churn.
  * SUPPORT-CONFIRMED RECENT-WINDOW present-set — an object is "present" only with >= ``min_support``
    detections within the last ``recent_window`` seconds, applied to a maintained table that is evicted
    after ``eviction_ttl_s``. This is the ghost-collapse fix: a scattered/transient false positive never
    accumulates recent support, so it never enters the present set — and stale objects are removed
    (``ObjectDB`` promotes by raw count and never removes from the permanent set).
  * CO-OCCURRENCE negative constraint: two detections the tracker says are DIFFERENT tracks (different
    ``track_id`` in the same frame) never collapse onto one identity — the sub-gate look-alike fix.

Real-data validation (Grounding-DINO + OWLv2 clips): look-alike merges 194->0, ghost present-set bounded
(3.73 vs 21.88 for an ObjectDB-like config), cross-session entry-time recovery via maintained identity.
Embedding is intentionally NOT used as an identity signal (real cosine ~0.77 between look-alikes is
non-discriminative); identity rides on track_id + position + the support/co-occurrence gates.

Time is driven by observation timestamps (``Object.ts``) when present, falling back to wall-clock — so the
present-set window and eviction behave correctly on both live streams and replayed recordings.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.perception.detection.type.detection3d.object import Object

logger = setup_logger()


def _ema(old: Vector3, new: Vector3, a: float) -> Vector3:
    """Exponential-moving-average of two positions (jitter/FP cannot teleport a track)."""
    return Vector3(
        a * new.x + (1.0 - a) * old.x,
        a * new.y + (1.0 - a) * old.y,
        a * new.z + (1.0 - a) * old.z,
    )


class WorldBelief:
    """Arch D engine, API-compatible with :class:`ObjectDB`.

    Drop-in: implements ``add_objects`` / ``get_objects`` / ``get_all_objects`` / ``promote`` /
    ``find_by_name`` / ``find_by_object_id`` / ``find_nearest`` / ``clear`` / ``get_stats`` /
    ``get_last_add_stats`` / ``agent_encode``. Constructor accepts the same tuning kwargs ObjectDB
    receives plus the Arch D knobs (with validated defaults).
    """

    def __init__(
        self,
        distance_threshold: float = 0.2,
        min_detections_for_permanent: int = 6,
        pending_ttl_s: float = 5.0,
        track_id_ttl_s: float = 5.0,
        *,
        min_support: int = 4,
        recent_window: float = 1.5,
        eviction_ttl_s: float = 60.0,
        pos_ema: float = 0.5,
        anchor_window: int = 9,          # SLAM-style robust landmark anchor: median over last N positions
        sticky_support: bool = True,
        label_gate: bool = True,
        cooccurrence_gate: bool = True,
        enable_history: bool = True,
        history_path: str | None = None,
        history_stream: str = "worldbelief_obs",
        reid_reacquire: bool = False,   # appearance re-acquisition: capability present but OFF by default
        reacq_window: float = 5.0,      # (validated as ineffective for identical-twin churn; see notes)
        reacq_radius: float = 0.30,
        reacq_cos: float = 0.55,
        reacq_margin: float = 0.08,
    ) -> None:
        self._distance_threshold = distance_threshold
        # ObjectDB-compat knobs (kept so the factory can pass them through unchanged):
        self._min_detections = min_detections_for_permanent
        self._track_id_ttl_s = track_id_ttl_s
        # Arch D knobs:
        self._min_support = int(min_support)
        self._recent_window = recent_window
        self._eviction_ttl_s = eviction_ttl_s          # must exceed expected absence of a re-entering object
        self._pos_ema = pos_ema
        self._anchor_window = int(anchor_window)
        self._sticky_support = sticky_support
        self._label_gate = label_gate
        self._cooccurrence_gate = cooccurrence_gate
        # appearance re-id: re-acquire a recently-DEPARTED identity (not a present one) when an object
        # reappears with no position match, by matching its appearance embedding. A margin guard keeps
        # identical look-alikes ambiguous (won't wrongly merge two cans). Needs detections to carry
        # `.embedding`; a no-op when none is present.
        self._reid_reacquire = reid_reacquire
        self._reacq_window = reacq_window      # only consider entities seen within this many seconds
        self._reacq_radius = reacq_radius      # ...and within this distance (object may have moved)
        self._reacq_cos = reacq_cos            # min cosine to accept a re-acquisition
        self._reacq_margin = reacq_margin      # best must beat 2nd-best by this (else ambiguous -> mint)

        self._entities: dict[str, Object] = {}         # one MAINTAINED table (object_id -> Object)
        self._meta: dict[str, dict[str, Any]] = {}     # object_id -> {entered_t,last_seen,support,window,labels}
        self._track_id_map: dict[int, str] = {}
        self._promoted: set[str] = set()               # objects force-kept present (e.g. select_object RPC)
        self._last_add_stats: dict[str, int] = {}
        self._now: float = 0.0
        self._lock = threading.RLock()

        # --- Phase 2: memory2 persistent history (when_entered / cross-session rehydrate) ---
        self._enable_history = enable_history
        self._history_path = history_path           # None -> a temp .db (auto); pass a path for cross-session
        self._history_stream_name = history_stream
        self._store: Any = None
        self._stream: Any = None

    def _ensure_history(self) -> None:
        """Lazily open the memory2 SqliteStore (real on-disk persistence -> survives a process restart)."""
        if not self._enable_history or self._stream is not None:
            return
        import tempfile

        from dimos.memory2.store.sqlite import SqliteStore
        path = self._history_path or tempfile.mktemp(suffix=".db", prefix="worldbelief_")
        self._history_path = path
        self._store = SqliteStore(path=path)
        self._store.start()
        self._stream = self._store.stream(self._history_stream_name, dict, codec="pickle")

    def _append_history(self, obj: Object, entered_t: float, now: float) -> None:
        if not self._enable_history:
            return
        self._ensure_history()
        c = obj.center
        self._stream.append(
            {"object_id": obj.object_id, "name": obj.name,
             "pos": [c.x, c.y, c.z] if c is not None else None, "entered_t": entered_t},
            ts=now, tags={"object_id": obj.object_id, "name": obj.name},
        )

    # ───────────────────────────── public API ─────────────────────────────

    def add_objects(self, objects: list[Object]) -> list[Object]:
        stats = {"input": len(objects), "created": 0, "updated": 0,
                 "matched_track": 0, "matched_distance": 0}
        results: list[Object] = []
        with self._lock:
            now = max((o.ts for o in objects if getattr(o, "ts", 0)), default=0.0) or time.time()
            self._now = now
            frame_claims: dict[str, int] = {}      # eid -> track_id claimed THIS frame (co-occurrence)
            for obj in objects:
                eid, reason = self._associate(obj, frame_claims)
                if eid is None:
                    new_obj = self._insert(obj, now)
                    results.append(new_obj)
                    stats["created"] += 1
                    # record the co-occurrence claim for the NEW entity too, so a second detection
                    # with a different track_id this frame cannot merge onto it.
                    if obj.track_id >= 0:
                        frame_claims[new_obj.object_id] = obj.track_id
                    continue
                self._update(eid, obj, now)
                results.append(self._entities[eid])
                stats["updated"] += 1
                stats["matched_track" if reason == "track" else "matched_distance"] += 1
                if obj.track_id >= 0:
                    frame_claims[eid] = obj.track_id
            self._evict_stale(now)
        stats["present"] = len(self.get_objects())
        stats["maintained"] = len(self._entities)
        self._last_add_stats = stats
        if stats["created"] > 0:
            logger.info(f"WorldBelief: {stats}")
        return results

    def get_last_add_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._last_add_stats)

    def get_objects(self) -> list[Object]:
        """Present-set = support-confirmed within the recent window (∪ force-promoted)."""
        with self._lock:
            cut = self._now - self._recent_window
            out: list[Object] = []
            for eid, m in self._meta.items():
                recent = sum(1 for t in m["window"] if t >= cut)
                if recent >= self._min_support or eid in self._promoted:
                    out.append(self._entities[eid])
            return out

    def get_all_objects(self) -> list[Object]:
        with self._lock:
            return list(self._entities.values())

    def promote(self, object_id: str) -> bool:
        with self._lock:
            if object_id in self._entities:
                self._promoted.add(object_id)
                return True
            return False

    def find_by_name(self, name: str) -> list[Object]:
        with self._lock:
            return [o for o in self.get_objects() if o.name == name]

    def find_by_object_id(self, object_id: str) -> Object | None:
        with self._lock:
            return self._entities.get(object_id)

    def find_nearest(self, position: Vector3, name: str | None = None) -> Object | None:
        with self._lock:
            cands = [o for o in self.get_objects()
                     if o.center is not None and (name is None or o.name == name)]
            if not cands:
                return None
            return min(cands, key=lambda o: position.distance(o.center))

    def clear(self) -> None:
        with self._lock:
            for obj in self._entities.values():
                obj.pointcloud = PointCloud2(
                    pointcloud=o3d.geometry.PointCloud(),
                    frame_id=obj.pointcloud.frame_id,
                    ts=obj.pointcloud.ts,
                )
            self._entities.clear()
            self._meta.clear()
            self._track_id_map.clear()
            self._promoted.clear()
            logger.info("WorldBelief cleared")

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            present = len(self.get_objects())
            return {"pending_count": len(self._entities) - present,
                    "permanent_count": present,
                    "total_count": len(self._entities)}

    def agent_encode(self) -> list[dict[str, Any]]:
        with self._lock:
            return [o.agent_encode() for o in self.get_objects()]

    # ───────────────────────────── association ─────────────────────────────

    @staticmethod
    def _median_center(positions) -> Vector3:
        """Component-wise median of recent positions = a drag-resistant landmark anchor."""
        m = np.median(np.asarray(positions, dtype=float), axis=0)
        return Vector3(float(m[0]), float(m[1]), float(m[2]))

    @staticmethod
    def _emb_of(obj: Object):
        """Unit-normalized appearance embedding of a detection, or None (a no-op signal)."""
        e = getattr(obj, "embedding", None)
        if e is None:
            return None
        v = np.asarray(e, dtype=np.float32)
        if v.size == 0:
            return None
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def _reacquire(self, obj: Object, frame_claims: dict[str, int]) -> str | None:
        """Re-acquire a recently-DEPARTED identity by appearance when an object reappears with no
        position match. Avoids stealing present ids (only entities absent > recent_window), and uses a
        best-vs-2nd-best margin so two identical cans stay ambiguous (mint new) rather than merge."""
        emb = self._emb_of(obj)
        if emb is None or obj.center is None:
            return None
        best, best_cos, second_cos = None, -1.0, -1.0
        for eid, e in self._entities.items():
            if frame_claims.get(eid) is not None:                 # claimed this frame -> not departed
                continue
            m = self._meta[eid]
            absent = self._now - m["last_seen"]
            if absent <= self._recent_window or absent > self._reacq_window:  # present, or too old
                continue
            if e.center is None or obj.center.distance(e.center) > self._reacq_radius:
                continue
            ce = m.get("emb")
            if ce is None:
                continue
            cos = float(np.dot(emb, ce))
            if cos > best_cos:
                best, best_cos, second_cos = eid, cos, best_cos
            elif cos > second_cos:
                second_cos = cos
        # instrumentation: count attempts that found a strong best but were BLOCKED by the margin guard
        # (i.e. an equally-good look-alike made it ambiguous) vs accepted re-acquisitions.
        if best is not None and best_cos >= self._reacq_cos:
            if (best_cos - second_cos) >= self._reacq_margin:
                self._reacq_accepted = getattr(self, "_reacq_accepted", 0) + 1
                return best
            self._reacq_blocked = getattr(self, "_reacq_blocked", 0) + 1
        return None

    def _associate(self, obj: Object, frame_claims: dict[str, int]) -> tuple[str | None, str | None]:
        """Sticky, support-prioritised association. Returns (object_id|None, reason)."""
        # priority 1: detector/tracker track_id (generous gate rejects stale-id reuse far away)
        if obj.track_id >= 0:
            eid = self._track_id_map.get(obj.track_id)
            if eid is not None and eid in self._entities:
                # co-occurrence MUST apply here too: if another track already took this entity THIS
                # frame, do not let track_id matching collapse a second detection onto it (the hole
                # that let a hand-wave poison _track_id_map -> chronic multi-detection merges).
                claimed = frame_claims.get(eid)
                blocked = self._cooccurrence_gate and claimed is not None and claimed != obj.track_id
                e = self._entities[eid]
                if not blocked and e.center is not None and obj.center is not None \
                        and obj.center.distance(e.center) <= 2 * self._distance_threshold \
                        and (self._now - self._meta[eid]["last_seen"]) <= self._track_id_ttl_s:
                    return eid, "track"
                if not blocked:
                    # stale/teleported id — forget the mapping, fall through to geometry
                    del self._track_id_map[obj.track_id]
                # if blocked, leave the mapping intact and fall through (this frame mints/re-routes)

        if obj.center is None:
            return None, None

        # priority 2: geometric NN, distance gate FIRST
        cands = [(obj.center.distance(e.center), eid, e)
                 for eid, e in self._entities.items()
                 if e.center is not None and obj.center.distance(e.center) <= self._distance_threshold]

        # co-occurrence: a tracked detection may NOT land on an entity another track already took this frame
        if self._cooccurrence_gate and obj.track_id >= 0 and cands:
            cands = [c for c in cands if frame_claims.get(c[1], obj.track_id) == obj.track_id]

        if not cands:
            # no position match -> try appearance re-acquisition of a recently-departed identity
            if self._reid_reacquire:
                reid = self._reacquire(obj, frame_claims)
                if reid is not None:
                    return reid, "reid"
            return None, None

        # soft label gate: prefer same-name candidates, fall back to all (label-swap robust)
        if self._label_gate:
            same = [c for c in cands if self._meta[c[1]]["labels"].most_common(1)[0][0] == obj.name]
            if same:
                cands = same

        if len(cands) == 1:
            return cands[0][1], "distance"

        # STICKY: prefer the established (highest-support) track, not merely the nearest; distance breaks ties
        if self._sticky_support:
            cands.sort(key=lambda c: (-self._meta[c[1]]["support"], c[0]))
            return cands[0][1], "distance"
        return min(cands, key=lambda c: c[0])[1], "distance"

    def _insert(self, obj: Object, now: float) -> Object:
        if not obj.ts:
            obj.ts = now
        self._entities[obj.object_id] = obj
        c = obj.center
        self._meta[obj.object_id] = {
            "entered_t": now, "last_seen": now, "support": 1,
            "window": [now], "labels": Counter([obj.name]), "emb": self._emb_of(obj),
            "positions": [[c.x, c.y, c.z]] if c is not None else [],
        }
        if obj.track_id >= 0:
            self._track_id_map[obj.track_id] = obj.object_id
        self._append_history(obj, now, now)
        return obj

    def _update(self, eid: str, obj: Object, now: float) -> None:
        existing = self._entities[eid]
        existing.update_object(obj)                    # accumulates pointcloud, latest size/pose, +1 count
        existing.ts = obj.ts or now
        m = self._meta[eid]
        # robust median ANCHOR (SLAM-style landmark): a transient occluder/crosser that briefly
        # associates is an OUTLIER the median rejects, so the anchor isn't dragged — a stationary
        # object keeps its true position through a crossing and re-matches on reappearance (no re-mint).
        if existing.center is not None:
            m["positions"].append([existing.center.x, existing.center.y, existing.center.z])
            if len(m["positions"]) > self._anchor_window:
                m["positions"] = m["positions"][-self._anchor_window:]
            existing.center = self._median_center(m["positions"])
        m["support"] += 1
        m["last_seen"] = now
        m["labels"][obj.name] += 1
        # report the STABLE majority-vote label, not the flapping latest detection — the open-vocab
        # detector flips a can between bottle/jar/can frame-to-frame; identity is stable, so the
        # reported label should be too (makes the validated label-robustness visible to consumers).
        existing.name = m["labels"].most_common(1)[0][0]
        e_emb = self._emb_of(obj)
        if e_emb is not None:
            m["emb"] = e_emb               # keep the latest appearance for re-acquisition matching
        m["window"].append(now)
        cut = now - self._recent_window
        if len(m["window"]) > 16 and m["window"][0] < cut:
            m["window"] = [t for t in m["window"] if t >= cut]
        if obj.track_id >= 0:
            self._track_id_map[obj.track_id] = eid
        self._append_history(existing, m["entered_t"], now)

    def _evict_stale(self, now: float) -> None:
        dead = [eid for eid, m in self._meta.items() if (now - m["last_seen"]) > self._eviction_ttl_s]
        for eid in dead:
            self._entities.pop(eid, None)
            self._meta.pop(eid, None)
            self._promoted.discard(eid)
            for tid, mapped in list(self._track_id_map.items()):
                if mapped == eid:
                    del self._track_id_map[tid]

    # ───────────────────────── temporal / cross-session ─────────────────────────

    def when_entered(self, object_id: str) -> float | None:
        """First-seen sim time of an object. Reads the maintained table if present, else the
        persisted history (so it answers even after the object left, and after a process restart)."""
        with self._lock:
            m = self._meta.get(object_id)
            if m is not None:
                return m["entered_t"]
            if not self._enable_history:
                return None
            self._ensure_history()
            best = None
            for obs in self._stream.tags(object_id=object_id):
                d = getattr(obs, "data", None) or obs._data
                et = d.get("entered_t", obs.ts) if isinstance(d, dict) else obs.ts
                best = et if best is None else min(best, et)
            return best

    def rehydrate(self) -> None:
        """Rebuild the maintained entity table FROM THE ON-DISK STORE (a true cross-process restore):
        preserves each object_id and its first-seen entered_t, so a returning object re-associates to
        its original identity and when_entered reports the genuine first sighting — not a fresh mint."""
        if not self._enable_history:
            return
        with self._lock:
            self._ensure_history()
            # build minimal Objects (geometry regenerates on next detection; identity+time is what we restore)
            import numpy as np
            import open3d as o3d  # type: ignore[import-untyped]

            from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
            from dimos.msgs.sensor_msgs.Image import Image
            from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
            from dimos.perception.detection.type.detection3d.object import Object

            img = Image(np.zeros((2, 2, 3), np.uint8))
            agg: dict[str, dict[str, Any]] = {}
            for obs in self._stream:
                d = getattr(obs, "data", None) or obs._data
                if not isinstance(d, dict):
                    continue
                oid = d.get("object_id")
                if oid is None or d.get("pos") is None:
                    continue
                a = agg.setdefault(oid, {"entered": d.get("entered_t", obs.ts), "last": obs.ts,
                                         "pos": d["pos"], "labels": Counter(), "n": 0})
                a["entered"] = min(a["entered"], d.get("entered_t", obs.ts))
                if obs.ts >= a["last"]:
                    a["last"], a["pos"] = obs.ts, d["pos"]
                a["labels"][d.get("name", "object")] += 1
                a["n"] += 1
            self._entities.clear(); self._meta.clear(); self._track_id_map.clear(); self._promoted.clear()
            for oid, a in agg.items():
                p = a["pos"]
                v = Vector3(p[0], p[1], p[2])
                pc = PointCloud2(pointcloud=o3d.geometry.PointCloud(), frame_id="world", ts=a["last"])
                name = a["labels"].most_common(1)[0][0]
                self._entities[oid] = Object(
                    object_id=oid, center=v, size=Vector3(0.05, 0.05, 0.1),
                    pose=PoseStamped(position=v), pointcloud=pc, image=img, bbox=(0, 0, 1, 1),
                    track_id=-1, class_id=0, confidence=0.5, name=name, ts=a["last"])
                self._meta[oid] = {"entered_t": a["entered"], "last_seen": a["last"], "support": a["n"],
                                   "window": [], "labels": a["labels"], "emb": None, "positions": [p]}
            self._now = max((a["last"] for a in agg.values()), default=0.0)
            logger.info(f"WorldBelief rehydrated {len(self._entities)} entities from {self._history_path}")

    def close(self) -> None:
        """Release the history store (call on shutdown)."""
        if self._store is not None:
            try:
                self._store.stop()
            except Exception:  # noqa: BLE001
                pass
            self._store = None
            self._stream = None

    def __len__(self) -> int:
        with self._lock:
            return len(self.get_objects())

    def __repr__(self) -> str:
        with self._lock:
            return f"WorldBelief(present={len(self.get_objects())}, maintained={len(self._entities)})"


def make_belief_engine(name: str = "world_belief", **kwargs: Any):
    """Factory: 'objectdb' (legacy) | 'world_belief' (Arch D, default). Both share ObjectDB's API."""
    if name == "objectdb":
        from dimos.perception.detection.objectDB import ObjectDB
        # ObjectDB takes only its four positional tuning kwargs
        ob_kw = {k: kwargs[k] for k in
                 ("distance_threshold", "min_detections_for_permanent", "pending_ttl_s", "track_id_ttl_s")
                 if k in kwargs}
        return ObjectDB(**ob_kw)
    if name == "world_belief":
        return WorldBelief(**kwargs)
    raise ValueError(f"unknown belief engine: {name!r} (use 'world_belief' or 'objectdb')")


__all__ = ["WorldBelief", "make_belief_engine"]
