r"""Real-time aerial detection -> ego-motion compensation -> tracking -> forecasting.

Pipeline (one pass per frame)
-----------------------------
1. ``Detector``             raw ``model.predict`` (verified hyper-parameters, native aspect ratio).
2. ``EgoMotionCompensator`` masked terrain flow -> homography ``H`` (prev frame -> current frame).
3. ``EntityTracker``        warps every track (trail + Kalman state) through ``H``, then
                            IoU -> centroid association. Every detection receives a track ID;
                            nothing is suppressed.
4. ``TrajectoryForecaster`` batched ``TrajectoryGRU`` on scale-normalised features, with a
                            constant-velocity fallback for young tracks / missing weights.
5. ``Visualizer``           boxes, IDs, ground-locked trails, forecast curves, 2-sigma ellipses.

Usage
-----
    python inference_pipeline.py --weights weights/detector.pt \
        --source input.mp4 --output output.mp4 --gru weights/trajectory_gru.pth
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from trajectory_common import (
    EgoMotionCompensator,
    ForecastConfig,
    TrajectoryGRU,
    build_features,
    load_checkpoint,
)

try:  # optimal assignment if SciPy is present (it ships with Ultralytics)
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - greedy fallback
    linear_sum_assignment = None


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Detections:
    """All detections of one frame. ``N`` may be zero."""

    boxes: np.ndarray       # (N, 4) float32 xyxy, native pixels
    confs: np.ndarray       # (N,)   float32
    cls_ids: np.ndarray     # (N,)   int64
    track_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


@dataclass(slots=True)
class Forecast:
    """Future waypoints for one track, in current-frame pixel coordinates."""

    waypoints: np.ndarray   # (K, 2)
    sigmas: np.ndarray      # (K, 2) std-dev in pixels (x, y)
    learned: bool           # True = TrajectoryGRU, False = constant-velocity fallback


class DetectorProtocol(Protocol):
    names: dict[int, str]

    def detect(self, frame: np.ndarray) -> Detections: ...


# --------------------------------------------------------------------------- #
# Stage 1 - detection
# --------------------------------------------------------------------------- #
class Detector:
    """YOLOv8 wrapper using the verified inference parameters (do not alter)."""

    CONF, IOU, IMGSZ, MAX_DET = 0.28, 0.45, 640, 25

    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO  # deferred so the tracker can be tested without it

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self.model.names)

    def detect(self, frame: np.ndarray) -> Detections:
        result = self.model.predict(
            source=frame,
            conf=self.CONF,
            iou=self.IOU,
            imgsz=self.IMGSZ,       # letterboxed internally: aspect ratio is preserved
            agnostic_nms=True,
            max_det=self.MAX_DET,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return Detections(
                np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
            )
        b = result.boxes
        return Detections(
            b.xyxy.cpu().numpy().astype(np.float32),
            b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(np.int64),
        )


# --------------------------------------------------------------------------- #
# Stage 3 - tracking
# --------------------------------------------------------------------------- #
_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
_HM = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


class Track:
    """Constant-velocity Kalman track living in *current-frame* pixel coordinates.

    State ``x = [cx, cy, vx, vy]``; because trails and state are re-warped through the
    ego-motion homography every frame, ``(vx, vy)`` is ground velocity in px/frame.
    """

    __slots__ = ("id", "cls_id", "x", "P", "size", "trail", "sizes", "hits", "misses", "max_len")

    def __init__(self, track_id: int, box: np.ndarray, cls_id: int, max_len: int) -> None:
        w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        s = 0.5 * (w + h)
        self.id = track_id
        self.cls_id = cls_id
        self.x = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]), 0.0, 0.0])
        self.P = np.diag([(0.1 * s) ** 2, (0.1 * s) ** 2, (0.5 * s) ** 2, (0.5 * s) ** 2])
        self.size = np.array([w, h], dtype=np.float64)
        self.trail = self.x[:2].reshape(1, 2).copy()     # (L, 2) filtered centres
        self.sizes = self.size.reshape(1, 2).copy()      # (L, 2) box sizes
        self.hits = 1
        self.misses = 0
        self.max_len = max_len

    # -- Kalman ---------------------------------------------------------------
    def predict(self) -> None:
        s = 0.5 * float(self.size.sum())
        q = np.diag([(0.04 * s) ** 2, (0.04 * s) ** 2, (0.02 * s) ** 2, (0.02 * s) ** 2])
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + q

    def update(self, box: np.ndarray, cls_id: int) -> None:
        w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        z = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
        r = np.eye(2) * (0.08 * 0.5 * (w + h)) ** 2
        S = _HM @ self.P @ _HM.T + r
        K = self.P @ _HM.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _HM @ self.x)
        self.P = (np.eye(4) - K @ _HM) @ self.P
        self.size = 0.6 * self.size + 0.4 * np.array([w, h])   # EMA against box jitter
        self.cls_id = cls_id
        self.hits += 1
        self.misses = 0
        self._append()

    def coast(self) -> None:
        """No detection this frame: keep the prediction so the trail stays contiguous."""
        self.misses += 1
        self._append()

    def _append(self) -> None:
        self.trail = np.vstack((self.trail, self.x[:2]))[-self.max_len:]
        self.sizes = np.vstack((self.sizes, self.size))[-self.max_len:]

    @property
    def box(self) -> np.ndarray:
        half = 0.5 * self.size
        return np.concatenate((self.x[:2] - half, self.x[:2] + half))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(M, 4)`` and ``(N, 4)`` xyxy boxes -> ``(M, N)``."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0.0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Min-cost one-to-one assignment, keeping only pairs with ``cost <= max_cost``."""
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
    pairs: list[tuple[int, int]] = []
    work = cost.copy()
    while True:
        r, c = np.unravel_index(int(np.argmin(work)), work.shape)
        if work[r, c] > max_cost:
            return pairs
        pairs.append((int(r), int(c)))
        work[r, :] = np.inf
        work[:, c] = np.inf


class EntityTracker:
    """Lossless IoU + Kalman/centroid tracker with ego-motion-warped state.

    Every detection passed to :meth:`update` is returned with a track ID: matched
    detections inherit the ID, unmatched ones open a new track immediately.
    """

    def __init__(
        self,
        iou_gate: float = 0.20,
        dist_gate: float = 1.0,
        max_misses: int = 20,
        trail_len: int = 90,
    ) -> None:
        self.iou_gate = iou_gate
        self.dist_gate = dist_gate          # centre distance in units of mean box diagonal
        self.max_misses = max_misses
        self.trail_len = trail_len
        self.tracks: list[Track] = []
        self._next_id = 1

    def _warp_tracks(self, H: np.ndarray) -> None:
        """Move all trails and Kalman states from previous-frame to current-frame coords."""
        if not self.tracks:
            return
        chunks: list[np.ndarray] = []
        for t in self.tracks:
            chunks += [t.trail, t.x[:2].reshape(1, 2), (t.x[:2] + t.x[2:]).reshape(1, 2)]
        warped = EgoMotionCompensator.warp_array(np.vstack(chunks), H)
        i = 0
        for t in self.tracks:
            n = len(t.trail)
            t.trail = warped[i:i + n]
            pos, tip = warped[i + n], warped[i + n + 1]
            t.x = np.concatenate((pos, tip - pos))      # velocity warped as a displacement
            i += n + 2

    def update(self, dets: Detections, H: np.ndarray) -> np.ndarray:
        """Advance one frame. Returns ``(N,)`` track IDs aligned with ``dets``."""
        self._warp_tracks(H)
        for t in self.tracks:
            t.predict()

        n_det = len(dets)
        ids = np.full(n_det, -1, dtype=np.int64)
        free_t = list(range(len(self.tracks)))
        free_d = list(range(n_det))

        if free_t and free_d:
            t_boxes = np.stack([t.box for t in self.tracks])
            d_boxes = dets.boxes.astype(np.float64)

            # Stage A: IoU
            iou = _iou_matrix(t_boxes, d_boxes)
            for r, c in _assign(1.0 - iou, 1.0 - self.iou_gate):
                self.tracks[r].update(d_boxes[c], int(dets.cls_ids[c]))
                ids[c] = self.tracks[r].id
                free_t.remove(r)
                free_d.remove(c)

            # Stage B: centroid distance (tiny / fast subjects with zero overlap)
            if free_t and free_d:
                tb, db = t_boxes[free_t], d_boxes[free_d]
                tc, dc = 0.5 * (tb[:, :2] + tb[:, 2:]), 0.5 * (db[:, :2] + db[:, 2:])
                diag_t = np.linalg.norm(tb[:, 2:] - tb[:, :2], axis=1)
                diag_d = np.linalg.norm(db[:, 2:] - db[:, :2], axis=1)
                dist = np.linalg.norm(tc[:, None] - dc[None], axis=2)
                dist /= np.maximum(0.5 * (diag_t[:, None] + diag_d[None]), 1e-6)
                matched_t, matched_d = [], []
                for r, c in _assign(dist, self.dist_gate):
                    ti, di = free_t[r], free_d[c]
                    self.tracks[ti].update(d_boxes[di], int(dets.cls_ids[di]))
                    ids[di] = self.tracks[ti].id
                    matched_t.append(ti)
                    matched_d.append(di)
                free_t = [i for i in free_t if i not in matched_t]
                free_d = [i for i in free_d if i not in matched_d]

        for ti in free_t:
            self.tracks[ti].coast()
        for di in free_d:
            track = Track(self._next_id, dets.boxes[di].astype(np.float64),
                          int(dets.cls_ids[di]), self.trail_len)
            self._next_id += 1
            self.tracks.append(track)
            ids[di] = track.id

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        dets.track_ids = ids
        return ids

    def visible_tracks(self) -> list[Track]:
        """Tracks matched to a detection in the current frame."""
        return [t for t in self.tracks if t.misses == 0]


# --------------------------------------------------------------------------- #
# Stage 4 - forecasting
# --------------------------------------------------------------------------- #
class TrajectoryForecaster:
    """Batched GRU forecasting with a constant-velocity fallback."""

    MIN_HITS_LINEAR = 4

    def __init__(self, weights: Path | None, device: torch.device) -> None:
        self.device = device
        self.model: TrajectoryGRU | None = None
        self.cfg = ForecastConfig()
        if weights is not None and weights.is_file():
            self.model = load_checkpoint(weights, device)
            self.cfg = self.model.cfg
        self._horizons = np.asarray(self.cfg.horizons, dtype=np.float64)

    @property
    def learned(self) -> bool:
        return self.model is not None

    def _linear(self, t: Track) -> Forecast:
        waypoints = t.x[:2][None] + self._horizons[:, None] * t.x[2:][None]
        growth = 0.15 + 0.6 * self._horizons / self._horizons[-1]
        return Forecast(waypoints, growth[:, None] * t.size[None], learned=False)

    @torch.inference_mode()
    def forecast(self, tracks: list[Track]) -> dict[int, Forecast]:
        """Return ``{track_id: Forecast}`` for every track with enough history."""
        out: dict[int, Forecast] = {}
        n_obs = self.cfg.num_observations
        ready = [t for t in tracks if self.model is not None and len(t.trail) >= n_obs]
        ready_ids = {t.id for t in ready}

        if ready:
            centers = np.stack([t.trail[-n_obs:] for t in ready])
            sizes = np.stack([t.sizes[-n_obs:] for t in ready])
            feats = torch.from_numpy(build_features(centers, sizes, self.cfg.feature_clip))
            mean, log_sigma = self.model(feats.to(self.device))
            mean = mean.cpu().numpy().astype(np.float64)
            sigma = np.exp(log_sigma.cpu().numpy().astype(np.float64))
            for i, t in enumerate(ready):
                scale = np.maximum(t.sizes[-1], 1.0)[None]
                out[t.id] = Forecast(t.trail[-1][None] + mean[i] * scale, sigma[i] * scale, True)

        for t in tracks:
            if t.id not in ready_ids and t.hits >= self.MIN_HITS_LINEAR:
                out[t.id] = self._linear(t)
        return out


# --------------------------------------------------------------------------- #
# Stage 5 - rendering
# --------------------------------------------------------------------------- #
class Visualizer:
    """Draws boxes, IDs, trails, forecast curves, uncertainty ellipses and the HUD."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, names: dict[int, str], horizons: tuple[int, ...], seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.names = names
        self.colors = {c: tuple(map(int, rng.integers(60, 240, size=3))) for c in names}
        self.horizons = np.asarray((0,) + tuple(horizons), dtype=np.float64)
        self._ts = np.linspace(0.0, float(horizons[-1]), 24)

    def _curve(self, origin: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
        """Smooth polyline through the current position and all waypoints."""
        pts = np.vstack((origin[None], waypoints))
        deg = min(2, len(pts) - 1)
        cx = np.polyfit(self.horizons, pts[:, 0], deg)
        cy = np.polyfit(self.horizons, pts[:, 1], deg)
        return np.stack((np.polyval(cx, self._ts), np.polyval(cy, self._ts)), axis=1)

    def render(
        self,
        frame: np.ndarray,
        dets: Detections,
        tracks: list[Track],
        forecasts: dict[int, Forecast],
        fps: float,
        ego_ok: bool,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        max_axis = max(h, w) // 4

        for t in tracks:
            color = self.colors.get(t.cls_id, (0, 255, 0))
            if len(t.trail) >= 2:
                cv2.polylines(frame, [np.round(t.trail).astype(np.int32)], False, color, 2, cv2.LINE_AA)
            fc = forecasts.get(t.id)
            if fc is None or not np.all(np.isfinite(fc.waypoints)):
                continue
            curve = np.clip(self._curve(t.x[:2], fc.waypoints), -4 * w, 4 * w)
            line_color = (0, 255, 255) if fc.learned else (200, 200, 200)
            cv2.polylines(frame, [np.round(curve).astype(np.int32)], False, line_color, 2, cv2.LINE_AA)
            for (px, py), (sx, sy) in zip(fc.waypoints, fc.sigmas):
                if not (-w < px < 2 * w and -h < py < 2 * h):
                    continue
                axes = (int(np.clip(2.0 * sx, 2, max_axis)), int(np.clip(2.0 * sy, 2, max_axis)))
                center = (int(round(px)), int(round(py)))
                cv2.ellipse(frame, center, axes, 0.0, 0.0, 360.0, line_color, 1, cv2.LINE_AA)
                cv2.circle(frame, center, 2, line_color, -1, cv2.LINE_AA)

        for box, conf, cls_id, tid in zip(dets.boxes, dets.confs, dets.cls_ids, dets.track_ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            color = self.colors.get(int(cls_id), (0, 255, 0))
            label = f"#{int(tid)} {self.names.get(int(cls_id), str(int(cls_id))).upper()} {int(conf * 100)}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, max(12, y1 - 3)), self.FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        counts = Counter(self.names.get(int(c), str(int(c))).upper() for c in dets.cls_ids)
        summary = " | ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "NO DETECTIONS"
        status = f"{fps:4.1f} FPS | EGO {'OK' if ego_ok else 'HOLD'}"
        cv2.rectangle(frame, (0, 0), (w, 36), (15, 15, 15), -1)
        cv2.putText(frame, f"DETECTED: {summary}", (12, 24), self.FONT, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(status, self.FONT, 0.55, 1)
        cv2.putText(frame, status, (w - sw - 12, 24), self.FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return frame


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class PipelineController:
    """Owns the frame loop and wires the single-responsibility stages together."""

    def __init__(
        self,
        detector: DetectorProtocol,
        ego: EgoMotionCompensator,
        tracker: EntityTracker,
        forecaster: TrajectoryForecaster,
        visualizer: Visualizer,
    ) -> None:
        self.detector = detector
        self.ego = ego
        self.tracker = tracker
        self.forecaster = forecaster
        self.visualizer = visualizer

    def process_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Run all stages on one BGR frame and return the annotated frame (in place)."""
        dets = self.detector.detect(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = self.ego.compute_homography(gray, dets.boxes)
        self.tracker.update(dets, H)
        visible = self.tracker.visible_tracks()
        forecasts = self.forecaster.forecast(visible)
        return self.visualizer.render(frame, dets, visible, forecasts, fps, self.ego.last_ok)

    def run(self, source: Path, output: Path, preview: bool = True) -> int:
        """Process a whole video. Returns the number of frames written."""
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video writer: {output}")

        preview_scale = min(1.0, 900.0 / max(width, height))
        frames, fps_ema, last = 0, 0.0, time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = self.process_frame(frame, fps_ema)
                writer.write(annotated)
                frames += 1

                now = time.perf_counter()
                inst = 1.0 / max(now - last, 1e-6)
                fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
                last = now

                if preview:
                    view = annotated if preview_scale == 1.0 else cv2.resize(
                        annotated, (int(width * preview_scale), int(height * preview_scale)))
                    cv2.imshow("Aerial Trajectory Forecasting", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if preview:
                cv2.destroyAllWindows()
        return frames


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=base / "weights" / "detector.pt")
    p.add_argument("--source", type=Path, default=base / "input.mp4")
    p.add_argument("--output", type=Path, default=base / "output.mp4")
    p.add_argument("--gru", type=Path, default=base / "weights" / "trajectory_gru.pth",
                   help="TrajectoryGRU checkpoint; constant-velocity fallback if missing")
    p.add_argument("--ego-max-dim", type=int, default=960,
                   help="longest side used for flow estimation (0 = native resolution)")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = Detector(args.weights)
    forecaster = TrajectoryForecaster(args.gru, device)
    print(f"Forecaster: {'TrajectoryGRU' if forecaster.learned else 'constant-velocity fallback'} "
          f"| horizons={forecaster.cfg.horizons} | device={device}")
    controller = PipelineController(
        detector=detector,
        ego=EgoMotionCompensator(max_dim=args.ego_max_dim or None),
        tracker=EntityTracker(),
        forecaster=forecaster,
        visualizer=Visualizer(detector.names, forecaster.cfg.horizons),
    )
    frames = controller.run(args.source, args.output, preview=not args.no_preview)
    print(f"Wrote {frames} frames -> {args.output}")


if __name__ == "__main__":
    main()
