"""Evaluate a trained TrajectoryGRU on the held-out validation sequences and plot results.

Rebuilds the same sequence-level validation split as ``train_trajectory_gru.py``
(same seed and dataset arguments), then writes four figures:

* ``eval_horizon_error.png``  mean error per horizon, GRU vs constant velocity
* ``eval_error_cdf.png``      error distribution at the longest horizon
* ``eval_calibration.png``    1-sigma / 2-sigma coverage of the predicted uncertainty
* ``eval_examples.png``       six example forecasts

    python tools/evaluate_model.py --checkpoint weights/trajectory_gru.pth \
        --visdrone-root <VID-train> <VID-val> --uavdt-images <...> --uavdt-gt <...>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _style import BLUE, INK, MUTED, ORANGE, apply_style  # noqa: E402
from train_trajectory_gru import build_split, load_uavdt, load_visdrone  # noqa: E402
from trajectory_common import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("weights/trajectory_gru.pth"))
    p.add_argument("--visdrone-root", type=Path, nargs="*", default=[])
    p.add_argument("--uavdt-images", type=Path, default=None)
    p.add_argument("--uavdt-gt", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("homography_cache"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--static-keep", type=float, default=0.25)
    p.add_argument("--ego-max-dim", type=int, default=960)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    seqs = []
    for root in args.visdrone_root:
        seqs += load_visdrone(root)
    if args.uavdt_images is not None and args.uavdt_gt is not None:
        seqs += load_uavdt(args.uavdt_images, args.uavdt_gt)
    if len(seqs) < 2:
        raise SystemExit("Need at least two annotated sequences (check dataset paths).")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * args.val_fraction)))
    val_seqs = [seqs[i] for i in order[:n_val]]

    model = load_checkpoint(args.checkpoint, torch.device("cpu"))
    cfg = model.cfg
    x_t, y_t = build_split(val_seqs, cfg, args, 0.0, rng, "val").tensors
    if len(x_t) == 0:
        raise SystemExit("No validation samples extracted.")
    with torch.inference_mode():
        means, logs = zip(*(model(x) for x in x_t.split(8192)))
    mean, sigma = torch.cat(means).numpy(), torch.cat(logs).exp().numpy()
    X, Y = x_t.numpy(), y_t.numpy()
    hz = np.asarray(cfg.horizons, dtype=np.float32)
    cv = X[:, -cfg.velocity_window:, :2].mean(axis=1, keepdims=True) * hz[None, :, None]
    err_gru = np.linalg.norm(mean - Y, axis=2)
    err_cv = np.linalg.norm(cv - Y, axis=2)

    print(f"Validation samples: {len(X)}")
    for i, h in enumerate(cfg.horizons):
        print(f"  t+{h}: GRU {err_gru[:, i].mean():.3f} | constant velocity {err_cv[:, i].mean():.3f}")

    # 1) mean error per horizon
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pos, bw = np.arange(len(hz)), 0.36
    for off, vals, color, name in ((-bw / 2, err_gru.mean(0), BLUE, "TrajectoryGRU"),
                                   (bw / 2, err_cv.mean(0), ORANGE, "Constant velocity")):
        bars = ax.bar(pos + off, vals, bw - 0.03, color=color, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=3, color=INK)
    ax.set_xticks(pos, [f"t+{int(h)}" for h in hz])
    ax.grid(axis="x", visible=False)
    ax.set(title="Mean error per horizon (lower is better)", ylabel="Error (box sizes)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out / "eval_horizon_error.png", dpi=150)

    # 2) error CDF at the longest horizon
    fig, ax = plt.subplots(figsize=(7, 4.2))
    q = np.linspace(0, 1, 400)
    for vals, color, name in ((err_gru[:, -1], BLUE, "TrajectoryGRU"),
                              (err_cv[:, -1], ORANGE, "Constant velocity")):
        ax.plot(np.quantile(vals, q), q * 100, color=color, lw=2, label=name)
    ax.set_xlim(0, float(np.quantile(err_cv[:, -1], 0.97)))
    ax.set(title=f"Error distribution at t+{int(hz[-1])} (further left is better)",
           xlabel="Error (box sizes)", ylabel="Share of samples (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "eval_error_cdf.png", dpi=150)

    # 3) uncertainty calibration
    z = np.abs(mean - Y) / sigma
    levels, expected = (1.0, 2.0), (68.3, 95.4)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bw = 0.8 / len(hz)
    for i, h in enumerate(hz):
        cover = [(z[:, i] <= k).mean() * 100 for k in levels]
        bars = ax.bar(np.arange(len(levels)) + (i - (len(hz) - 1) / 2) * bw, cover, bw - 0.03,
                      color=BLUE, alpha=0.45 + 0.55 * i / max(1, len(hz) - 1), label=f"t+{int(h)}")
        ax.bar_label(bars, fmt="%.0f", padding=3, color=INK)
    for j, e in enumerate(expected):
        ax.hlines(e, j - 0.45, j + 0.45, color=INK, lw=1.5, linestyles="--")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK, lw=1.5, ls="--"))
    labels.append("ideal (68.3% / 95.4%)")
    ax.set_xticks(range(len(levels)), ["within 1 sigma", "within 2 sigma"])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 108)
    ax.set(title="Uncertainty calibration (closer to ideal is better)", ylabel="Coverage (%)")
    ax.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(args.out / "eval_calibration.png", dpi=150, bbox_inches="tight")

    # 4) example forecasts
    moving = np.where(np.linalg.norm(Y[:, -1], axis=1) > 1.0)[0]
    pool = moving if len(moving) >= 6 else np.arange(len(X))
    pick = np.random.default_rng(0).choice(pool, min(6, len(pool)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, i in zip(axes.ravel(), pick):
        hist = np.vstack(([0, 0], -np.cumsum(X[i, ::-1, :2], axis=0)))[::-1]
        ax.plot(hist[:, 0], hist[:, 1], color=MUTED, lw=2, label="History")
        ax.plot(*np.vstack(([0, 0], Y[i])).T, "o-", color=INK, lw=1.5, ms=5, label="Ground truth")
        ax.plot(*np.vstack(([0, 0], cv[i])).T, "s--", color=ORANGE, lw=1.5, ms=5, label="Constant velocity")
        ax.plot(*np.vstack(([0, 0], mean[i])).T, "o-", color=BLUE, lw=2, ms=6, label="TrajectoryGRU")
        for (mx, my), (sx, sy) in zip(mean[i], sigma[i]):
            ax.add_patch(Ellipse((mx, my), 4 * sx, 4 * sy, fill=False, color=BLUE, lw=1, alpha=0.6))
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"sample {i}: GRU {err_gru[i, -1]:.2f} | const-vel {err_cv[i, -1]:.2f}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Example forecasts (units: box sizes, ellipse = 2 sigma, title = t+{int(hz[-1])} error)")
    fig.tight_layout()
    fig.savefig(args.out / "eval_examples.png", dpi=150)
    print(f"Saved four eval_*.png figures to {args.out}")


if __name__ == "__main__":
    main()
