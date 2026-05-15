"""LazyPerceptionModule — agent-callable open-vocab perception.

Three @skill methods, each a one-line composition of memory2 query
primitives. See ``spec.py`` for the architecture docstring.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import numpy as np

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.stream import Out
from dimos.manipulation.memory2.scene_store import SceneStore
from dimos.manipulation.memory2.spec import LazyPerceptionModuleConfig
from dimos.memory2.module import MemoryModule
from dimos.models.embedding.base import EmbeddingModel
from dimos.models.vl.base import VlModel
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.detection.type.detection3d.object import Object as DetObject
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _relative_time(ts: float) -> str:
    """Format a past timestamp as a human-readable age."""
    delta = max(0.0, time.time() - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}min ago"
    return f"{int(delta // 3600)}h ago"


class LazyPerceptionModule(MemoryModule):
    """Lazy memory2-native open-vocab object detector.

    Three skills, each a one-line memory2 composition. Stateless:
    every call is an independent query → VLM (find_objects only)
    → 3D → publish.
    """

    config: LazyPerceptionModuleConfig

    objects: Out[list[DetObject]]

    _vlm: VlModel | None = None
    _clip: EmbeddingModel | None = None
    _scene: SceneStore | None = None
    # object_id -> latest full DetObject (with geometry) seen THIS session.
    # The scene store persists identity/timestamps cross-session; geometry for
    # Meshcat/planner only exists for objects actually captured this run.
    _obj_cache: dict[str, DetObject]

    @rpc
    def start(self) -> None:
        super().start()
        self._vlm = self.register_disposable(self.config.vlm_provider())
        self._vlm.start()
        self._clip = self.register_disposable(self.config.embedding_model())
        self._clip.start()
        self._scene = SceneStore(self.store, match_distance=self.config.scene_match_distance)
        self._obj_cache = {}

    # ------------------------------------------------------------------ skills

    @skill
    def find_objects(self, prompt: str) -> str:
        """Find objects matching ``prompt``. Returns most recent confident
        match with timestamp.

        Open-vocab: ``prompt`` can be any natural language. Comma-separated
        prompts are split and processed per class because Moondream's
        ``query_detections`` labels every result with the literal query
        string.
        """
        prompt_list = [p.strip() for p in prompt.split(",") if p.strip()]
        if not prompt_list:
            self._scene_publish([])  # keep the existing scene; don't wipe Meshcat
            return "No prompts provided."

        all_objects: list[DetObject] = []
        for single in prompt_list:
            objs, _ = self._find_and_project(
                single,
                build_query=lambda stream, vec: stream.search(vec),
            )
            all_objects.extend(objs)

        most_recent_ts = self._scene_publish(all_objects)
        if not all_objects:
            return f"No confident '{prompt}' match in memory."

        age = _relative_time(most_recent_ts) if most_recent_ts is not None else "just now"
        lines = [self._fmt_object_line(o) for o in all_objects]
        return f"Found {len(all_objects)} object(s) matching '{prompt}' (seen {age}):\n" + "\n".join(lines)

    @skill
    def find_objects_near(
        self,
        prompt: str,
        x: float,
        y: float,
        z: float,
        radius: float = 1.0,
    ) -> str:
        """Find objects matching ``prompt`` in frames recorded when the
        camera was within ``radius`` meters of ``(x, y, z)``.

        ``.near()`` filters by the camera's pose at record time, NOT by
        the detected object's position. Note: applied as a Python
        post-filter after vector search; R*Tree pre-gating of the
        vector index is a memory2 follow-up.
        """
        prompt_list = [p.strip() for p in prompt.split(",") if p.strip()]
        if not prompt_list:
            self._scene_publish([])  # keep the existing scene; don't wipe Meshcat
            return "No prompts provided."

        pose = (x, y, z)
        all_objects: list[DetObject] = []
        for single in prompt_list:
            objs, _ = self._find_and_project(
                single,
                build_query=lambda stream, vec: stream.near(pose, radius).search(vec),
            )
            all_objects.extend(objs)

        most_recent_ts = self._scene_publish(all_objects)
        if not all_objects:
            return f"No confident '{prompt}' match near ({x:.2f}, {y:.2f}, {z:.2f}) within {radius}m."

        age = _relative_time(most_recent_ts) if most_recent_ts is not None else "just now"
        lines = [self._fmt_object_line(o) for o in all_objects]
        return (
            f"Found {len(all_objects)} object(s) matching '{prompt}' "
            f"near ({x:.2f}, {y:.2f}, {z:.2f}) within {radius}m (seen {age}):\n"
            + "\n".join(lines)
        )

    @skill
    def recall(self, name: str) -> str:
        """When and where did I last actually see something matching ``name``?

        Reads the persisted scene model (VLM-confirmed object sightings), NOT
        raw CLIP frame matches — so "(seen N ago)" reflects the true last time
        the object was genuinely present, ages correctly, and works across
        process restarts (the scene stream persists in the SQLite store).
        No VLM/CLIP call: pure lookup, cheap.
        """
        if self._scene is None:
            return f"No memory of '{name}'."
        try:
            rec = self._scene.last_seen(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("recall(%r) failed: %s", name, e)
            return f"No memory of '{name}'."

        if rec is None:
            return f"No memory of '{name}'."
        age = _relative_time(rec.last_seen)
        return (
            f"Last saw '{rec.name}' at ({rec.x:.2f}, {rec.y:.2f}, {rec.z:.2f}) "
            f"({age}); seen {rec.count}x total."
        )

    # ----------------------------------------------------------- internal

    def _scene_publish(self, new_objects: list[DetObject]) -> float | None:
        """Upsert VLM-confirmed detections into the persisted scene, cache
        their geometry for this session, then publish the FULL active scene
        — not just this query's hits. This is what fixes both symptoms:

        - recall reads the scene's last_seen (truthful, cross-session).
        - Publishing the full active set (instead of the per-prompt slice)
          makes the full-replace obstacle/Meshcat consumer keep every known
          object instead of evicting everything outside the current prompt.

        Returns the most-recent last_seen among ``new_objects`` (for the
        "(seen N ago)" line — ≈now since they were just VLM-confirmed).
        """
        most_recent: float | None = None
        if self._scene is None:
            try:
                self.objects.publish(list(new_objects))
            except Exception as e:  # noqa: BLE001
                logger.warning("publish failed: %s", e)
            return None

        for det in new_objects:
            c = det.center
            try:
                rec = self._scene.upsert(
                    det.name, float(c.x), float(c.y), float(c.z)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("scene upsert failed (%s): %s", det.name, e)
                continue
            det.object_id = rec.object_id
            self._obj_cache[rec.object_id] = det
            if most_recent is None or rec.last_seen > most_recent:
                most_recent = rec.last_seen

        # Full active scene → only objects whose geometry we captured this
        # session can be drawn/planned-against; identity/last_seen for the
        # rest still lives in the persisted store for recall.
        try:
            active = self._scene.current_scene(self.config.scene_ttl_s)
            publish = [
                self._obj_cache[o.object_id]
                for o in active
                if o.object_id in self._obj_cache
            ]
            self.objects.publish(publish)
        except Exception as e:  # noqa: BLE001
            logger.warning("scene publish failed: %s", e)
        return most_recent

    def _find_and_project(
        self,
        prompt: str,
        build_query: Callable[[Any, Any], Any],
    ) -> tuple[list[DetObject], float | None]:
        """Composed memory2 pipeline for ONE prompt class.

        Returns (objects, observation_ts). ``build_query`` is a callable
        ``(stream, query_vec) -> filtered Stream`` so find_objects /
        find_objects_near can share the rest of the pipeline.
        """
        if self._clip is None:
            return [], None
        try:
            vec = self._clip.embed_text(prompt)
            obs = (
                build_query(self.store.streams.color_image_embedded, vec)
                    .filter(lambda o: (o.similarity or 0) >= self.config.min_similarity)
                    .order_by("ts", desc=True)
                    .first()
            )
        except (AttributeError, LookupError):
            return [], None
        except Exception as e:
            logger.warning("_find_and_project(%r): %s", prompt, e)
            return [], None

        return self._detect_and_project_one(obs, prompt), obs.ts

    def _detect_and_project_one(self, color_obs: Any, prompt: str) -> list[DetObject]:
        """VLM detection + 3D projection for ONE peak frame, ONE prompt class."""
        if self._vlm is None:
            return []

        # Aligned depth + latest intrinsics. .first()/.last() raise LookupError on empty.
        try:
            depth_obs = self.store.streams.depth_image.at(
                color_obs.ts, tolerance=0.1
            ).first()
            info_obs = self.store.streams.camera_info.last()
        except LookupError:
            logger.warning("missing depth/info near ts=%.3f", color_obs.ts)
            return []
        except AttributeError:
            logger.warning("depth_image or camera_info stream not yet available")
            return []

        # VLM can hang (network) or OOM (CUDA). Treat failure as no-detection.
        try:
            dets_2d = self._vlm.query_detections(color_obs.data, prompt)
        except Exception as e:
            logger.warning("VLM failed (prompt=%r ts=%.3f): %s", prompt, color_obs.ts, e)
            return []
        if not dets_2d.detections:
            return []

        # RealSense publishes depth as DEPTH16 (uint16 millimeters). The recorder
        # persists it raw; from_2d_to_list expects meters (depth_scale=1.0). The
        # live OSR path converts mm->m inline (object_scene_registration.py:295-301)
        # but the memory2-native path replays raw recorded DEPTH16, so we must do
        # the same conversion here or every point exceeds depth_trunc and projects
        # to nothing. Format-aware: only DEPTH16 is divided.
        depth_img = depth_obs.data
        depth_cv = depth_img.to_opencv()
        if depth_img.format == ImageFormat.DEPTH16:
            depth_cv = depth_cv.astype(np.float32) / 1000.0
        elif depth_cv.dtype != np.float32:
            depth_cv = depth_cv.astype(np.float32)

        # Foreground depth clustering. Moondream returns a rectangular bbox
        # (no segmentation mask), so from_2d_to_list's rectangular depth mask
        # captures the object PLUS background; the AABB center then sits
        # ~halfway to the wall (measured ~0.24m bias — the arm plans to empty
        # space). Per bbox: histogram the depths, keep the NEAREST DOMINANT
        # cluster at its actual extent (grown until a real empty gap to the
        # background). Adapts to object size + camera angle + distance; no
        # fixed band. Heuristic limit: object touching the wall has no gap.
        depth_cv = depth_cv.copy()
        h, w = depth_cv.shape[:2]
        for det in dets_2d.detections:
            try:
                bx1, by1, bx2, by2 = (int(v) for v in det.bbox)
            except (TypeError, ValueError):
                continue
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            roi = depth_cv[by1:by2, bx1:bx2]
            bounds = self._nearest_cluster_bounds(roi[(roi > 0.05) & np.isfinite(roi)])
            if bounds is None:
                continue
            near, far = bounds
            roi[(roi < near) | (roi > far)] = 0.0

        depth_m = Image(
            data=depth_cv,
            format=ImageFormat.DEPTH,
            frame_id=depth_img.frame_id,
            ts=depth_img.ts,
        )

        camera_transform = self._camera_transform_from_pose(color_obs.pose)
        try:
            # from_2d_to_list's annotation says ImageDetections2D[Detection2DSeg]
            # but the implementation at object.py:205-215 handles bbox-only
            # detections too (synthesizes a rectangular mask). Moondream returns
            # bbox-only — works at runtime.
            return DetObject.from_2d_to_list(
                detections_2d=dets_2d,  # type: ignore[arg-type]
                color_image=color_obs.data,
                depth_image=depth_m,
                camera_info=info_obs.data,
                camera_transform=camera_transform,
                max_distance=self.config.max_distance,
                use_aabb=self.config.use_aabb,
                max_obstacle_width=self.config.max_obstacle_width,
            )
        except Exception as e:
            logger.warning("from_2d_to_list failed at ts=%.3f: %s", color_obs.ts, e)
            return []

    def _nearest_cluster_bounds(self, depths: Any) -> tuple[float, float] | None:
        """Nearest dominant depth cluster within a bbox, at its actual extent.

        Histograms the bbox's valid depths, finds the closest occupied bin, and
        grows the cluster forward until an empty span >= ``foreground_gap``
        (the object↔background separation). Returns (near, far) meter bounds, or
        ``None`` to skip filtering (too few points / degenerate). No fixed band:
        the cluster keeps whatever depth extent it actually has, so it adapts to
        object size, camera viewing angle, and distance.
        """
        if depths.size < self.config.foreground_min_points:
            return None
        lo = float(np.percentile(depths, 1))   # drop near speckle
        hi = float(np.percentile(depths, 99))  # drop far flyers
        if hi <= lo:
            return None
        bin_size = self.config.foreground_bin_size
        nbins = max(1, int(np.ceil((hi - lo) / bin_size)))
        hist, edges = np.histogram(depths, bins=nbins, range=(lo, hi))
        # "Occupied" = holds a meaningful fraction of points; filters sparse
        # noise bins that would otherwise bridge object→background.
        occ_thresh = max(3, int(0.005 * depths.size))
        occupied = hist >= occ_thresh
        if not occupied.any():
            return None
        first = int(np.argmax(occupied))
        gap_bins = max(1, int(np.ceil(self.config.foreground_gap / bin_size)))
        end = first
        empty_run = 0
        for b in range(first, nbins):
            if occupied[b]:
                end = b
                empty_run = 0
            else:
                empty_run += 1
                if empty_run >= gap_bins:
                    break
        return float(edges[first]), float(edges[end + 1])

    def _camera_transform_from_pose(self, pose: Any) -> Transform | None:
        """Build a Transform from the recorder's pose-stamped observation.

        Observation.pose is always a 7-tuple (x, y, z, qx, qy, qz, qw) or None
        (memory2/observationstore/sqlite.py:_decompose_pose).
        Object.from_2d_to_list(camera_transform=T) applies T to a camera-frame
        pointcloud to produce world-frame output — T is camera→world; the
        recorder's pose-in-world IS that. NO .inverse().
        """
        if pose is None:
            return None
        try:
            x, y, z, qx, qy, qz, qw = pose
        except (TypeError, ValueError):
            logger.warning("unexpected pose shape: %r", type(pose).__name__)
            return None
        return Transform(
            translation=Vector3(x, y, z),
            rotation=Quaternion(qx, qy, qz, qw),
        )

    @staticmethod
    def _fmt_object_line(o: DetObject) -> str:
        return f"  - {o.name} at ({o.center.x:.2f}, {o.center.y:.2f}, {o.center.z:.2f})"
