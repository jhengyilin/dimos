"""Memory2-persisted scene/object model.

The raw memory2 streams (color/depth/embedded) are an append-only sensor log
with no notion of object identity or "is this object actually present". That
is why ``recall`` reported a perpetually-fresh "1s ago" (it ranked any
weakly-CLIP-matching frame by recency) and why detections vanished from Meshcat
when you queried something else (the per-prompt publisher fed a full-replace
consumer).

This module adds the missing layer: a derived, **persisted** stream of
VLM-confirmed object sightings with stable identity and timestamps — ObjectDB's
proven model (spatial dedup, last/first seen, detection count) but written into
memory2 instead of RAM, so it survives process restarts (cross-session memory).

Design:
- Append-only ``objects_scene`` stream. One record per genuine VLM-confirmed
  detection. Never pruned → ``recall`` can answer "last saw X 3h ago" across
  sessions by querying the full history.
- Identity: spatial-proximity dedup on upsert (ObjectDB's ``_match_by_distance``
  ported). Same place ⇒ same ``object_id``; ``last_seen`` advances only on a
  real re-detection.
- ``current_scene(ttl)`` reduces the log to latest-record-per-object within a
  freshness window — the *active* scene published to Meshcat/the planner.
  ``recall`` ignores TTL (full history); the scene snapshot uses it so stale
  objects age out of the collision world.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import uuid
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class SceneObject:
    """One object's current best estimate in the persisted scene.

    Stored as the payload of an ``objects_scene`` observation. Plain dataclass
    so memory2's PickleCodec round-trips it (no lcm_encode needed).
    """

    object_id: str
    name: str
    x: float
    y: float
    z: float
    first_seen: float
    last_seen: float
    count: int

    def distance_to(self, x: float, y: float, z: float) -> float:
        return ((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2) ** 0.5


class SceneStore:
    """Wraps a memory2 ``objects_scene`` stream with identity-aware upsert.

    Not a Module — a helper owned by LazyPerceptionModule, using its store so
    the scene persists in the same SQLite file as the recordings.
    """

    STREAM = "objects_scene"

    def __init__(self, store: Any, match_distance: float = 0.15) -> None:
        self._store = store
        self._match_distance = match_distance
        self._last_ts = 0.0  # strictly-increasing append ts (ts column is UNIQUE)

    # -- internal -----------------------------------------------------------

    def _stream(self) -> Any:
        # Created on first access; reused (and reloaded) across sessions because
        # memory2 persists the stream + its registry entry.
        return self._store.stream(self.STREAM, SceneObject)

    def _next_ts(self, now: float) -> float:
        # memory2's per-stream ts column is NOT NULL UNIQUE; guarantee strictly
        # increasing so multiple objects from one detection frame don't collide.
        ts = now if now > self._last_ts else self._last_ts + 1e-4
        self._last_ts = ts
        return ts

    def _recent_records(self, since: float | None) -> list[SceneObject]:
        """All scene records (optionally after ``since``), newest first."""
        try:
            s = self._stream()
            q = s.after(since) if since is not None else s
            return [o.data for o in q.order_by("ts", desc=True)]
        except (AttributeError, LookupError):
            return []  # stream not created yet (no detections this session)
        except Exception as e:  # noqa: BLE001
            logger.warning("scene _recent_records failed: %s", e)
            return []

    def _latest_per_object(self, records: list[SceneObject]) -> dict[str, SceneObject]:
        """records are newest-first → first occurrence of each id is its latest."""
        latest: dict[str, SceneObject] = {}
        for r in records:
            if r.object_id not in latest:
                latest[r.object_id] = r
        return latest

    # -- write --------------------------------------------------------------

    def upsert(self, name: str, x: float, y: float, z: float, now: float | None = None) -> SceneObject:
        """Record a VLM-confirmed sighting. Spatial-dedup against the active
        scene: same place ⇒ same object_id, advance last_seen; else new id.
        """
        now = now if now is not None else time.time()
        active = self._latest_per_object(self._recent_records(since=None)).values()
        match = min(
            (o for o in active if o.distance_to(x, y, z) < self._match_distance),
            key=lambda o: o.distance_to(x, y, z),
            default=None,
        )
        if match is not None:
            rec = SceneObject(
                object_id=match.object_id,
                name=name or match.name,
                x=x, y=y, z=z,
                first_seen=match.first_seen,
                last_seen=now,
                count=match.count + 1,
            )
        else:
            rec = SceneObject(
                object_id=uuid.uuid4().hex[:8],
                name=name,
                x=x, y=y, z=z,
                first_seen=now,
                last_seen=now,
                count=1,
            )
        try:
            self._stream().append(
                rec,
                ts=self._next_ts(now),
                pose=(x, y, z, 0.0, 0.0, 0.0, 1.0),
                tags={"object_id": rec.object_id, "name": rec.name},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("scene upsert append failed (%s): %s", name, e)
        return rec

    # -- read ---------------------------------------------------------------

    def last_seen(self, name: str) -> SceneObject | None:
        """Most recent sighting whose name fuzzy-matches ``name`` across ALL
        history (no TTL — answers cross-session "when did I last see X").
        """
        nm = name.strip().lower()
        for r in self._recent_records(since=None):  # newest first
            rn = (r.name or "").lower()
            if nm and (nm in rn or rn in nm):
                return r
        return None

    def current_scene(self, ttl_s: float) -> list[SceneObject]:
        """Latest record per object whose last_seen is within ``ttl_s`` — the
        active scene to publish to the planner/Meshcat. Stale objects age out.
        """
        cutoff = time.time() - ttl_s if ttl_s and ttl_s > 0 else None
        records = self._recent_records(since=cutoff)
        latest = self._latest_per_object(records)
        if cutoff is None:
            return list(latest.values())
        return [o for o in latest.values() if o.last_seen >= cutoff]
