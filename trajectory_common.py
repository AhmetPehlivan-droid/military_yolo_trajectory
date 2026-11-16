"""Shared building blocks for the aerial trajectory forecasting system.

Both ``inference_pipeline.py`` and ``train_trajectory_gru.py`` import from this
module so that ego-motion compensation, feature construction and the network
definition are byte-for-byte identical at training and inference time.

Contents
--------
* ``EgoMotionCompensator`` - masked Shi-Tomasi + pyramidal LK + RANSAC homography.
* ``ForecastConfig``       - hyper-parameters stored inside the checkpoint.
* ``build_features``       - scale-normalised (dx/w, dy/h, dw/w, dh/h) features.
* ``TrajectoryGRU``        - compact GRU with mean + log-sigma heads.
* ``gaussian_nll``         - heteroscedastic regression loss.
* ``save_checkpoint`` / ``load_checkpoint``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import nn

BoxXYXY = tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Ego-motion compensation
# --------------------------------------------------------------------------- #
class EgoMotionCompensator:
    """Estimates the frame-to-frame ground-plane homography of a moving camera.

    ``compute_homography`` returns ``H`` (3x3, float64) such that a pixel ``p``
    in the *previous* frame maps to ``H @ p`` in the *current* frame.

    Parameters
    ----------
    max_corners:
        Maximum number of Shi-Tomasi terrain corners.
    pad:
        Padding (native pixels) added around every foreground box in the mask.
    max_dim:
        If set, flow is estimated on a copy whose longest side is at most
        ``max_dim`` (aspect ratio preserved) and ``H`` is rescaled back to
        native pixel units. ``None`` runs at native resolution. This only
        affects ego-motion estimation, never the detector input.
    ransac_thresh:
        RANSAC reprojection threshold in native pixels.
    """

    MIN_CORNERS = 12
    MIN_TRACKED = 10

    def __init__(
        self,
        max_corners: int = 400,
        pad: int = 12,
        max_dim: int | None = 960,
        ransac_thresh: float = 4.0,
    ) -> None:
        self.pad = pad
        self.max_dim = max_dim
        self.ransac_thresh = ransac_thresh
        self.prev_gray: np.ndarray | None = None
        self._prev_boxes: np.ndarray = np.empty((0, 4), dtype=np.float64)
        self.last_inliers: int = 0
        self.last_ok: bool = False
        self.feature_params = dict(
            maxCorners=max_corners, qualityLevel=0.01, minDistance=15, blockSize=7
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self) -> None:
        """Forget the previous frame (call at sequence boundaries / scene cuts)."""
        self.prev_gray = None
        self._prev_boxes = np.empty((0, 4), dtype=np.float64)
        self.last_inliers = 0
        self.last_ok = False

    def compute_homography(
        self, curr_gray: np.ndarray, exclude_boxes: Iterable[Sequence[float]]
    ) -> np.ndarray:
        """Return H mapping previous-frame pixels to current-frame pixels."""
        identity = np.eye(3, dtype=np.float64)
        boxes = np.asarray(list(exclude_boxes), dtype=np.float64).reshape(-1, 4)

        scale = 1.0
        if self.max_dim is not None:
            longest = max(curr_gray.shape[:2])
            if longest > self.max_dim:
                scale = self.max_dim / float(longest)
        work = (
            cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale != 1.0
            else curr_gray
        )

        prev_gray, prev_boxes = self.prev_gray, self._prev_boxes
        self.prev_gray, self._prev_boxes = work, boxes
        self.last_inliers, self.last_ok = 0, False

        if prev_gray is None or prev_gray.shape != work.shape:
            return identity

        # Corners are sampled on the previous frame, so the subjects must be
        # masked where they were (previous boxes) and where they are now.
        mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
        h, w = mask.shape
        for x1, y1, x2, y2 in np.vstack((prev_boxes, boxes)):
            p1 = (max(0, int((x1 - self.pad) * scale)), max(0, int((y1 - self.pad) * scale)))
            p2 = (min(w, int((x2 + self.pad) * scale)), min(h, int((y2 + self.pad) * scale)))
            cv2.rectangle(mask, p1, p2, 0, -1)

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
        if p0 is None or len(p0) < self.MIN_CORNERS:
            return identity

        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, work, p0, None, **self.lk_params)
        if p1 is None or status is None:
            return identity
        ok = status.reshape(-1) == 1
        good_old = p0.reshape(-1, 2)[ok]
        good_new = p1.reshape(-1, 2)[ok]
        if len(good_old) < self.MIN_TRACKED:
            return identity

        thresh = max(1.5, self.ransac_thresh * scale)
        H, inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, thresh)
        if H is None or inliers is None or not np.all(np.isfinite(H)):
            return identity

        H = H.astype(np.float64)
        if scale != 1.0:  # work-space -> native pixel units
            S = np.diag([scale, scale, 1.0])
            H = np.linalg.inv(S) @ H @ S
        if abs(H[2, 2]) < 1e-9:
            return identity
        H /= H[2, 2]

        # Reject physically implausible frame-to-frame warps.
        det = float(np.linalg.det(H[:2, :2]))
        n_inliers = int(inliers.sum())
        if not (0.5 < det < 2.0) or n_inliers < self.MIN_TRACKED:
            return identity

        self.last_inliers, self.last_ok = n_inliers, True
        return H

    @staticmethod
    def warp_array(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp an ``(N, 2)`` array through ``H``; returns ``(N, 2)`` float64."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)

    @staticmethod
    def transform_points(
        points_list: Sequence[tuple[float, float]], H: np.ndarray
    ) -> list[tuple[float, float]]:
        """List-of-tuples variant of :meth:`warp_array` (original interface)."""
        if len(points_list) == 0:
            return []
        warped = EgoMotionCompensator.warp_array(np.asarray(points_list), H)
        return [(float(x), float(y)) for x, y in warped]


# --------------------------------------------------------------------------- #
# Forecaster configuration + features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForecastConfig:
    """Hyper-parameters that must match between training and inference."""

    history_len: int = 15                       # feature steps (needs history_len + 1 observations)
    horizons: tuple[int, ...] = (15, 30, 45)    # future frames to predict
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    feature_clip: float = 5.0
    velocity_window: int = 5                    # steps averaged for the constant-velocity prior

    @property
    def num_observations(self) -> int:
        return self.history_len + 1


def build_features(centers: np.ndarray, sizes: np.ndarray, clip: float) -> np.ndarray:
    """Scale-normalised motion features.

    Parameters
    ----------
    centers : ``(N, T + 1, 2)`` ego-motion-compensated box centres, all expressed
        in the coordinate system of the last observation's frame.
    sizes   : ``(N, T + 1, 2)`` box ``(w, h)`` at each observation.

    Returns
    -------
    ``(N, T, 4)`` float32: ``[dx/w_t, dy/h_t, (w_t - w_{t-1})/w_{t-1}, (h_t - h_{t-1})/h_{t-1}]``.
    """
    sizes = np.maximum(np.asarray(sizes, dtype=np.float64), 1.0)
    centers = np.asarray(centers, dtype=np.float64)
    feats = np.empty((centers.shape[0], centers.shape[1] - 1, 4), dtype=np.float64)
    feats[..., 0:2] = (centers[:, 1:] - centers[:, :-1]) / sizes[:, 1:]
    feats[..., 2:4] = (sizes[:, 1:] - sizes[:, :-1]) / sizes[:, :-1]
    return np.clip(feats, -clip, clip).astype(np.float32)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class TrajectoryGRU(nn.Module):
    """GRU encoder with a probabilistic multi-horizon displacement head.

    The head predicts a *residual* on top of a constant-velocity extrapolation of the
    last ``velocity_window`` steps, so an untrained network equals the kinematic
    baseline and training only has to learn turns, braking and scattering.

    Input  : ``(B, history_len, 4)`` features from :func:`build_features`.
    Output : ``mean``      ``(B, K, 2)`` - displacement from the last observed
             centre to each horizon, in units of the last box ``(w, h)``;
             ``log_sigma`` ``(B, K, 2)`` - log std-dev in the same units.
    """

    LOG_SIGMA_RANGE = (-5.0, 3.0)

    def __init__(self, cfg: ForecastConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_horizons = len(cfg.horizons)
        self.gru = nn.GRU(
            input_size=4,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_size, self.num_horizons * 4),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer(
            "horizon_steps",
            torch.tensor(cfg.horizons, dtype=torch.float32).view(1, -1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_n = self.gru(x)
        out = self.head(h_n[-1]).view(-1, self.num_horizons, 4)
        velocity = x[:, -self.cfg.velocity_window:, :2].mean(dim=1, keepdim=True)
        mean = velocity * self.horizon_steps + out[..., :2]
        log_sigma = out[..., 2:].clamp(*self.LOG_SIGMA_RANGE)
        return mean, log_sigma


def gaussian_nll(mean: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean diagonal-Gaussian negative log-likelihood (constant term dropped)."""
    return (log_sigma + 0.5 * ((target - mean) * torch.exp(-log_sigma)) ** 2).mean()


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model: TrajectoryGRU, extra: dict | None = None) -> None:
    """Save weights + config as plain tensors/builtins (``weights_only`` safe)."""
    cfg = asdict(model.cfg)
    cfg["horizons"] = list(cfg["horizons"])
    payload = {"state_dict": model.state_dict(), "config": cfg, "extra": extra or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: Path, device: torch.device) -> TrajectoryGRU:
    """Rebuild a :class:`TrajectoryGRU` in eval mode from ``save_checkpoint`` output."""
    payload = torch.load(str(path), map_location=device, weights_only=True)
    cfg_dict = dict(payload["config"])
    cfg_dict["horizons"] = tuple(int(h) for h in cfg_dict["horizons"])
    model = TrajectoryGRU(ForecastConfig(**cfg_dict))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
