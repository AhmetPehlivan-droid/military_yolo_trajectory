r"""Dataset extraction + training for ``TrajectoryGRU`` on VisDrone-VID and UAVDT.

Per sequence
------------
1. Run ``EgoMotionCompensator`` over every frame (ground-truth boxes masked, pad=12)
   to obtain ``H_k`` (frame k-1 -> frame k). Homographies are cached on disk.
2. For each anchor frame ``t``, chain the homographies so that the 16 observations
   ``t-15 .. t`` and the future frames ``t+15, t+30, t+45`` of every track are
   expressed in the coordinate system of frame ``t`` - exactly what the tracker's
   per-frame trail warping produces at inference.
3. Features : ``[dx/w, dy/h, dw/w, dh/h]`` for 15 steps        -> ``(15, 4)``
   Targets  : ``(future_centre - centre_t) / (w_t, h_t)``      -> ``(3, 2)``

Expected layouts
----------------
VisDrone-VID split root (pass one or more, e.g. train and val)::

    <root>/sequences/<seq>/0000001.jpg ...
    <root>/annotations/<seq>.txt     frame,id,x,y,w,h,score,category,trunc,occl

UAVDT::

    <images>/<seq>/img000001.jpg ...
    <gt>/<seq>_gt_whole.txt          frame,id,x,y,w,h,out_of_view,occlusion,category

Usage
-----
    python train_trajectory_gru.py \
        --visdrone-root <VisDrone2019-VID-train> <VisDrone2019-VID-val> \
        --uavdt-images <UAV-benchmark-M> \
        --uavdt-gt <UAV-benchmark-MOTD_v1.0/GT> \
        --output weights/trajectory_gru.pth
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from trajectory_common import (
    EgoMotionCompensator,
    ForecastConfig,
    TrajectoryGRU,
    build_features,
    gaussian_nll,
    save_checkpoint,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VISDRONE_VALID_CATEGORIES = tuple(range(1, 11))   # 0 = ignored region, 11 = others


# --------------------------------------------------------------------------- #
# Sequence discovery
# --------------------------------------------------------------------------- #
@dataclass
class SequenceSpec:
    """One annotated video sequence.

    ``annotations`` is ``(M, 6)`` float64: ``[frame_pos, track_id, x, y, w, h]`` where
    ``frame_pos`` indexes ``frame_paths`` (0-based) and ``(x, y)`` is the top-left corner.
    """

    name: str
    frame_paths: list[Path]
    annotations: np.ndarray


def _frame_number(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    if not digits:
        raise ValueError(f"No frame number in file name: {path.name}")
    return int(digits[-1])


def _build_sequence(name: str, frame_dir: Path, raw: np.ndarray) -> SequenceSpec | None:
    """Map annotation frame numbers onto sorted image files; ``raw`` = [frame,id,x,y,w,h]."""
    frames = sorted((p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                    key=_frame_number)
    if not frames or raw.size == 0:
        return None
    lookup = {_frame_number(p): i for i, p in enumerate(frames)}
    pos = np.array([lookup.get(int(f), -1) for f in raw[:, 0]], dtype=np.float64)
    keep = (pos >= 0) & (raw[:, 4] > 1) & (raw[:, 5] > 1)
    if not keep.any():
        return None
    ann = raw[keep].copy()
    ann[:, 0] = pos[keep]
    return SequenceSpec(name, frames, ann)


def _read_csv(path: Path, n_cols: int) -> np.ndarray:
    data = np.genfromtxt(str(path), delimiter=",", usecols=range(n_cols), dtype=np.float64)
    return np.atleast_2d(data) if data.size else np.empty((0, n_cols))


def load_visdrone(root: Path) -> list[SequenceSpec]:
    seq_root, ann_root = root / "sequences", root / "annotations"
    if not seq_root.is_dir() or not ann_root.is_dir():
        raise FileNotFoundError(f"{root} must contain 'sequences/' and 'annotations/'")
    specs: list[SequenceSpec] = []
    for ann_file in sorted(ann_root.glob("*.txt")):
        frame_dir = seq_root / ann_file.stem
        if not frame_dir.is_dir():
            continue
        raw = _read_csv(ann_file, 8)
        raw = raw[(raw[:, 6] > 0) & np.isin(raw[:, 7], VISDRONE_VALID_CATEGORIES)]
        spec = _build_sequence(f"visdrone_{ann_file.stem}", frame_dir, raw[:, :6])
        if spec is not None:
            specs.append(spec)
    return specs


def load_uavdt(images_root: Path, gt_root: Path) -> list[SequenceSpec]:
    if not images_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(f"UAVDT paths not found: {images_root} / {gt_root}")
    specs: list[SequenceSpec] = []
    for gt_file in sorted(gt_root.glob("*_gt_whole.txt")):
        seq = gt_file.name.replace("_gt_whole.txt", "")
        frame_dir = images_root / seq
        if not frame_dir.is_dir():
            continue
        spec = _build_sequence(f"uavdt_{seq}", frame_dir, _read_csv(gt_file, 6))
        if spec is not None:
            specs.append(spec)
    return specs


# --------------------------------------------------------------------------- #
# Ego-motion + sample extraction
# --------------------------------------------------------------------------- #
def compute_homographies(seq: SequenceSpec, cache_dir: Path, ego_max_dim: int | None) -> np.ndarray:
    """Return ``(n_frames, 3, 3)``; ``H[k]`` maps frame ``k-1`` to frame ``k`` (``H[0] = I``)."""
    cache = cache_dir / f"{seq.name}_H{ego_max_dim or 0}.npy"
    n = len(seq.frame_paths)
    if cache.is_file():
        cached = np.load(str(cache))
        if cached.shape == (n, 3, 3):
            return cached

    order = np.argsort(seq.annotations[:, 0], kind="stable")
    ann = seq.annotations[order]
    starts = np.searchsorted(ann[:, 0], np.arange(n + 1))
    xyxy = np.column_stack((ann[:, 2], ann[:, 3], ann[:, 2] + ann[:, 4], ann[:, 3] + ann[:, 5]))

    ego = EgoMotionCompensator(max_dim=ego_max_dim)
    H = np.tile(np.eye(3), (n, 1, 1))
    for k, path in enumerate(seq.frame_paths):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [warn] unreadable frame skipped: {path}")
            ego.reset()
            continue
        H[k] = ego.compute_homography(gray, xyxy[starts[k]:starts[k + 1]])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache), H)
    return H


def extract_samples(
    seq: SequenceSpec,
    H: np.ndarray,
    cfg: ForecastConfig,
    stride: int,
    jitter: float,
    static_keep: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``X (N, history_len, 4)`` and ``Y (N, K, 2)`` for one sequence."""
    n_frames = len(seq.frame_paths)
    hist, horizons, max_h = cfg.history_len, cfg.horizons, max(cfg.horizons)
    ann = seq.annotations
    _, row = np.unique(ann[:, 1], return_inverse=True)
    col = ann[:, 0].astype(np.int64)

    dense = np.full((int(row.max()) + 1, n_frames, 4), np.nan, dtype=np.float64)
    dense[row, col, 0] = ann[:, 2] + 0.5 * ann[:, 4]
    dense[row, col, 1] = ann[:, 3] + 0.5 * ann[:, 5]
    dense[row, col, 2] = ann[:, 4]
    dense[row, col, 3] = ann[:, 5]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    fut_idx = [h for h in horizons]
    for t in range(hist, n_frames - max_h, stride):
        obs = dense[:, t - hist:t + 1]
        fut = dense[:, [t + h for h in fut_idx], :2]
        valid = ~np.isnan(obs).any(axis=(1, 2)) & ~np.isnan(fut).any(axis=(1, 2))
        if not valid.any():
            continue
        obs, fut = obs[valid], fut[valid]
        centers = np.empty_like(obs[..., :2])
        sizes = obs[..., 2:].copy()

        # History: T maps frame k -> anchor frame t.
        T = np.eye(3)
        for j in range(hist, -1, -1):
            centers[:, j] = EgoMotionCompensator.warp_array(obs[:, j, :2], T)
            T = T @ H[t - hist + j]
            T /= T[2, 2]

        # Future: F maps anchor frame t -> frame k, so its inverse brings k back to t.
        fut_c = np.empty_like(fut)
        F = np.eye(3)
        ok = True
        for k in range(t + 1, t + max_h + 1):
            F = H[k] @ F
            F /= F[2, 2]
            if (k - t) in fut_idx:
                if abs(np.linalg.det(F)) < 1e-9:
                    ok = False
                    break
                i = fut_idx.index(k - t)
                fut_c[:, i] = EgoMotionCompensator.warp_array(fut[:, i], np.linalg.inv(F))
        if not ok:
            continue

        clean_anchor = centers[:, -1].copy()
        if jitter > 0.0:  # emulate detector box noise on the observed history only
            centers = centers + rng.normal(0.0, jitter, centers.shape) * sizes
            sizes = sizes * (1.0 + rng.normal(0.0, jitter, sizes.shape))
            sizes = np.maximum(sizes, 1.0)

        x = build_features(centers, sizes, cfg.feature_clip)
        y = (fut_c - centers[:, -1][:, None]) / np.maximum(sizes[:, -1], 1.0)[:, None]

        clean_disp = np.linalg.norm((fut_c[:, -1] - clean_anchor) / np.maximum(obs[:, -1, 2:], 1.0), axis=1)
        keep = np.isfinite(y).all(axis=(1, 2)) & (np.abs(y).max(axis=(1, 2)) < 60.0)
        static = clean_disp < 0.25
        keep &= ~static | (rng.random(len(y)) < static_keep)   # parked / resting subjects dominate
        if keep.any():
            xs.append(x[keep])
            ys.append(y[keep].astype(np.float32))

    k = len(horizons)
    if not xs:
        return np.empty((0, hist, 4), np.float32), np.empty((0, k, 2), np.float32)
    return np.concatenate(xs), np.concatenate(ys)


def build_split(
    seqs: list[SequenceSpec], cfg: ForecastConfig, args: argparse.Namespace,
    jitter: float, rng: np.random.Generator, label: str,
) -> TensorDataset:
    xs, ys = [], []
    for i, seq in enumerate(seqs, 1):
        H = compute_homographies(seq, args.cache_dir, args.ego_max_dim or None)
        x, y = extract_samples(seq, H, cfg, args.stride, jitter, args.static_keep, rng)
        print(f"  [{label} {i}/{len(seqs)}] {seq.name}: {len(seq.frame_paths)} frames -> {len(x)} samples")
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs) if xs else np.empty((0, cfg.history_len, 4), np.float32)
    y_all = np.concatenate(ys) if ys else np.empty((0, len(cfg.horizons), 2), np.float32)
    return TensorDataset(torch.from_numpy(x_all), torch.from_numpy(y_all))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(model: TrajectoryGRU, loader: DataLoader, device: torch.device,
             horizons: tuple[int, ...]) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (NLL, per-horizon GRU error, per-horizon constant-velocity error).

    Errors are mean Euclidean distances in units of box size (w, h).
    """
    model.eval()
    hz = torch.tensor(horizons, dtype=torch.float32, device=device).view(1, -1, 1)
    nll_sum, n = 0.0, 0
    err_gru = torch.zeros(len(horizons), device=device)
    err_cv = torch.zeros(len(horizons), device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, log_sigma = model(x)
        nll_sum += gaussian_nll(mean, log_sigma, y).item() * len(x)
        err_gru += (mean - y).norm(dim=2).sum(dim=0)
        cv = x[:, -model.cfg.velocity_window:, :2].mean(dim=1, keepdim=True) * hz
        err_cv += (cv - y).norm(dim=2).sum(dim=0)
        n += len(x)
    n = max(n, 1)
    return nll_sum / n, (err_gru / n).cpu().numpy(), (err_cv / n).cpu().numpy()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ForecastConfig(
        history_len=args.history, horizons=tuple(args.horizons),
        hidden_size=args.hidden_size, num_layers=args.num_layers,
    )

    seqs: list[SequenceSpec] = []
    for root in args.visdrone_root:
        seqs += load_visdrone(root)
    if args.uavdt_images is not None and args.uavdt_gt is not None:
        seqs += load_uavdt(args.uavdt_images, args.uavdt_gt)
    if len(seqs) < 2:
        raise SystemExit("Need at least two annotated sequences (check dataset paths).")

    # Split by sequence so validation tracks are never seen during training.
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * args.val_fraction)))
    val_seqs = [seqs[i] for i in order[:n_val]]
    train_seqs = [seqs[i] for i in order[n_val:]]
    print(f"Sequences: {len(train_seqs)} train / {len(val_seqs)} val | device={device}")

    train_ds = build_split(train_seqs, cfg, args, args.jitter, rng, "train")
    val_ds = build_split(val_seqs, cfg, args, 0.0, rng, "val")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("No samples extracted; tracks may be shorter than history + horizon.")
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = TrajectoryGRU(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.epochs * len(train_loader))

    best = float("inf")
    history: list[dict] = []
    history_path = args.output.with_suffix(".history.json")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mean, log_sigma = model(x)
            loss = gaussian_nll(mean, log_sigma, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item() * len(x)
            seen += len(x)

        val_nll, err_gru, err_cv = evaluate(model, val_loader, device, cfg.horizons)
        fmt = lambda a: " ".join(f"t+{h}:{v:.3f}" for h, v in zip(cfg.horizons, a))
        flag = ""
        if val_nll < best:
            best = val_nll
            save_checkpoint(args.output, model, {"val_nll": val_nll, "epoch": epoch})
            flag = "  [saved]"
        history.append({
            "epoch": epoch, "train_nll": running / seen, "val_nll": val_nll,
            "gru_err": {str(h): float(v) for h, v in zip(cfg.horizons, err_gru)},
            "const_vel_err": {str(h): float(v) for h, v in zip(cfg.horizons, err_cv)},
        })
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch:3d} | train NLL {running / seen:7.4f} | val NLL {val_nll:7.4f} | "
              f"GRU err {fmt(err_gru)} | const-vel err {fmt(err_cv)}{flag}")

    print(f"Best val NLL {best:.4f} -> {args.output} | history -> {history_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visdrone-root", type=Path, nargs="*", default=[],
                   help="VisDrone-VID split roots containing sequences/ and annotations/")
    p.add_argument("--uavdt-images", type=Path, default=None, help="UAVDT image root (UAV-benchmark-M)")
    p.add_argument("--uavdt-gt", type=Path, default=None, help="UAVDT GT dir with *_gt_whole.txt")
    p.add_argument("--output", type=Path, default=Path("trajectory_gru.pth"))
    p.add_argument("--cache-dir", type=Path, default=Path("homography_cache"))
    p.add_argument("--history", type=int, default=15)
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 45])
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--stride", type=int, default=3, help="anchor-frame stride")
    p.add_argument("--jitter", type=float, default=0.03,
                   help="std of simulated detector noise on training history (fraction of box size)")
    p.add_argument("--static-keep", type=float, default=0.25,
                   help="fraction of near-stationary samples kept")
    p.add_argument("--ego-max-dim", type=int, default=960,
                   help="must match inference_pipeline.py (0 = native resolution)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
