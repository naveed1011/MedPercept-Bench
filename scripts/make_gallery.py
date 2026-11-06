#!/usr/bin/env python3
"""Save side-by-side GT-vs-prediction figures from REAL predictions (qualitative results).

Usage: python scripts/make_gallery.py --n 8
Writes results/figures/gallery_*.png - used for the failure/success gallery in the README.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.datasets import KvasirSegDataset, BCCDDataset  # noqa: E402
from src.visualize import save_side_by_side  # noqa: E402


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _mask(b64, shape):
    m = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    if m.size != (shape[1], shape[0]):
        m = m.resize((shape[1], shape[0]))
    return np.array(m) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "datasets.yaml"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pred_dir = os.path.join(REPO_ROOT, cfg.get("output_root", "results"), "predictions")
    fig_dir = os.path.join(REPO_ROOT, cfg.get("output_root", "results"), "figures")
    data_root = os.path.join(REPO_ROOT, cfg.get("data_root", "data"))

    kv = KvasirSegDataset(root=data_root, subset=cfg["kvasir"]["subset"], seed=cfg["kvasir"]["seed"])
    det = _load(os.path.join(pred_dir, "kvasir_detection.json"))
    seg = _load(os.path.join(pred_dir, "kvasir_seg_polyp.json"))
    saved = 0
    for i in range(len(kv)):
        if saved >= args.n:
            break
        it = kv[i]
        iid = it["image_id"]
        d, s = det.get(iid), seg.get(iid)
        if not d and not s:
            continue
        shape = (it["height"], it["width"])
        pm = [_mask(m, shape) for m in (s or {}).get("masks_b64_png", [])]
        save_side_by_side(it["image"], it["gt_boxes"], it["gt_labels"], it["gt_masks"],
                          np.asarray((d or {}).get("boxes_xyxy", [])).reshape(-1, 4),
                          (d or {}).get("labels", []), pm,
                          os.path.join(fig_dir, f"gallery_kvasir_{iid}.png"), title=iid)
        saved += 1
    print(f"Saved {saved} gallery figures to {fig_dir}")
    if saved == 0:
        print("No real predictions found - run scripts/run_inference.py first.")


if __name__ == "__main__":
    main()
