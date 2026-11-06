"""Visualization: save prediction-overlay + GT side-by-side figures."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def _draw_boxes(ax, boxes, labels, color="red", score=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        lab = labels[i] if i < len(labels) else ""
        if lab:
            ax.text(x1, max(y1 - 5, 0), lab, color=color, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"))


def _overlay_masks(ax, image: np.ndarray, masks, color=(255, 0, 0), alpha=0.4):
    out = image.copy().astype(np.float32)
    for m in masks:
        m = np.asarray(m).astype(bool)
        if m.shape != out.shape[:2]:
            continue
        out[m] = out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ax.imshow(out.astype(np.uint8))


def save_side_by_side(
    image: Image.Image,
    gt_boxes: np.ndarray,
    gt_labels: list,
    gt_masks: list,
    pred_boxes: np.ndarray,
    pred_labels: list,
    pred_masks: list,
    out_path: str,
    title: str = "",
):
    """Save a 2-panel figure: GT (left) vs Prediction (right)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = np.array(image.convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # GT panel
    if gt_masks:
        _overlay_masks(axes[0], img, gt_masks, color=(0, 200, 0))
    else:
        axes[0].imshow(img)
    _draw_boxes(axes[0], np.asarray(gt_boxes).reshape(-1, 4), gt_labels, color="green")
    axes[0].set_title(f"Ground Truth{(' — ' + title) if title else ''}")
    axes[0].axis("off")
    # Pred panel
    if pred_masks:
        _overlay_masks(axes[1], img, pred_masks, color=(255, 80, 80))
    else:
        axes[1].imshow(img)
    _draw_boxes(axes[1], np.asarray(pred_boxes).reshape(-1, 4), pred_labels, color="red")
    axes[1].set_title("Prediction (WeVisOne-VL zero-shot)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar_chart(metric_dict: dict, out_path: str, title: str = "MedPercept-Bench metrics"):
    """Save a bar chart of key metrics. metric_dict: {label: value}."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = list(metric_dict.keys())
    values = [float(v) for v in metric_dict.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
