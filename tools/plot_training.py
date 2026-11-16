"""Plot training curves from a ``*.history.json`` file or a pasted console log.

    python tools/plot_training.py weights/trajectory_gru.history.json --out docs
    python tools/plot_training.py train_log.txt --out docs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from _style import BLUE, INK, ORANGE, apply_style

_ROW = re.compile(
    r"epoch\s+(\d+)\s*\|\s*train NLL\s+(-?[\d.]+)\s*\|\s*val NLL\s+(-?[\d.]+)\s*\|"
    r"\s*GRU err (.*?)\|\s*const-vel err (.*?)(\[saved\])?\s*$", re.M)
_PAIR = re.compile(r"t\+(\d+):([\d.]+)")


def load_history(path: Path) -> list[dict]:
    """Return a list of per-epoch records from JSON history or console text."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    records = [
        {
            "epoch": int(m.group(1)), "train_nll": float(m.group(2)), "val_nll": float(m.group(3)),
            "gru_err": {h: float(v) for h, v in _PAIR.findall(m.group(4))},
            "const_vel_err": {h: float(v) for h, v in _PAIR.findall(m.group(5))},
        }
        for m in _ROW.finditer(text)
    ]
    if not records:
        raise SystemExit(f"No epoch lines found in {path}")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    hist = load_history(args.history)
    epochs = [r["epoch"] for r in hist]
    val = [r["val_nll"] for r in hist]
    best = val.index(min(val))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(epochs, [r["train_nll"] for r in hist], color=BLUE, lw=2, label="Train NLL")
    ax.plot(epochs, val, color=ORANGE, lw=2, label="Validation NLL")
    ax.scatter([epochs[best]], [val[best]], s=70, color=ORANGE, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    ax.annotate(f"best: epoch {epochs[best]} ({val[best]:.3f})", (epochs[best], val[best]),
                textcoords="offset points", xytext=(8, 12), color=INK)
    ax.set(title="Loss curve (lower is better)", xlabel="Epoch", ylabel="Gaussian NLL")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "training_loss.png", dpi=150)

    horizons = sorted(hist[0]["gru_err"], key=int)
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.8), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        gru = [r["gru_err"][h] for r in hist]
        cv = [r["const_vel_err"][h] for r in hist]
        ax.plot(epochs, gru, color=BLUE, lw=2, label="TrajectoryGRU")
        ax.plot(epochs, cv, color=ORANGE, lw=2, label="Constant velocity")
        gain = 100.0 * (1.0 - gru[best] / cv[best])
        ax.set(title=f"t+{h} frames ({gain:.0f}% better at best epoch)", xlabel="Epoch")
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("Mean error (box sizes)")
    axes[0][0].legend(frameon=False, loc="lower left")
    fig.suptitle("Validation error: GRU vs constant-velocity baseline", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out / "training_error.png", dpi=150, bbox_inches="tight")
    print(f"Saved training_loss.png and training_error.png to {args.out}")


if __name__ == "__main__":
    main()
