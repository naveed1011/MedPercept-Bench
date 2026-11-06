#!/usr/bin/env python3
"""Run WeVisOne-VL (or mock) inference on dataset subsets and dump predictions.

Outputs:
    results/predictions/<dataset>_<task>.json   aggregated predictions
    results/predictions/per_image/*.json         one file per image
    results/predictions/masks/*.png              segmentation masks (if any)

Usage:
    python scripts/run_inference.py --config configs/datasets.yaml
    python scripts/run_inference.py --mock        # mock predictor, synthetic data
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.model_wrapper import ModelWrapper  # noqa: E402
from src.datasets import KvasirSegDataset, BCCDDataset, SyntheticDataset  # noqa: E402


def _mask_to_png_b64(mask: np.ndarray) -> str:
    m = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pred_to_json(pred: dict, image_id: str, task: str, query) -> dict:
    return {
        "image_id": image_id,
        "task": task,
        "query": query if isinstance(query, str) else list(query),
        "boxes_xyxy": np.asarray(pred["boxes_xyxy"]).tolist(),
        "labels": list(pred["labels"]),
        "scores": [float(s) for s in pred["scores"]],
        "masks_b64_png": [_mask_to_png_b64(m) for m in pred.get("masks", [])],
        "count": int(pred["count"]),
        "rejected": bool(pred["rejected"]),
        "raw_response": pred["raw_response"],
        "latency_s": float(pred["latency_s"]),
    }


def run_inference_on_dataset(model: ModelWrapper, dataset, task: str, queries,
                             out_dir: str, tag: str, max_images: Optional[int] = None,
                             max_new_tokens: int = 1024) -> dict:
    """Run inference for one (task, queries) pair.

    Crash-safe by design (this is what keeps Colab runs alive):
      * every image is written to per_image/<tag>_<id>.json as soon as it is done
      * re-running the script RESUMES: images that already have a per-image file are skipped
      * a GPU out-of-memory on one image is recorded as error="oom" and the run continues
      * GPU memory is released every few images
    """
    per_image_dir = os.path.join(out_dir, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)
    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    if n == 0:
        raise RuntimeError(
            f"[{tag}] dataset is EMPTY - nothing to run. Did scripts/download_data.sh succeed? "
            "Check that data/ contains Kvasir-SEG/ and BCCD_Dataset-master/.")

    predictions, n_done, n_resumed, n_oom = {}, 0, 0, 0
    for i in range(n):
        item = dataset[i]
        image_id = item["image_id"]
        rec_path = os.path.join(per_image_dir, f"{tag}_{image_id}.json")
        if os.path.exists(rec_path):  # resume
            with open(rec_path) as f:
                predictions[image_id] = json.load(f)
            n_resumed += 1
            continue
        pred = model.predict(item["image"], task=task, query=queries, image_id=image_id,
                             max_new_tokens=max_new_tokens)
        rec = _pred_to_json(pred, image_id, task, queries)
        if pred.get("error"):
            rec["error"] = pred["error"]
            n_oom += 1
            # do NOT write a per-image file for failed images, so a re-run retries them
        else:
            with open(rec_path, "w") as f:
                json.dump(rec, f)
        predictions[image_id] = rec
        n_done += 1
        if n_done % 5 == 0 and hasattr(model, "_free_gpu"):
            model._free_gpu()
        if n_done % 10 == 0 or i == n - 1:
            print(f"    [{tag}] {i + 1}/{n}", flush=True)

    agg_path = os.path.join(out_dir, f"{tag}.json")
    with open(agg_path, "w") as f:
        json.dump(predictions, f)
    msg = f"  [{tag}] {len(predictions)} images ({n_resumed} resumed, {n_oom} OOM) -> {agg_path}"
    print(msg)
    return predictions


def build_oracle_gt(dataset) -> dict:
    """Build oracle GT dict for the mock predictor (mock mode only)."""
    oracle = {}
    for i in range(len(dataset)):
        item = dataset[i]
        oracle[item["image_id"]] = {
            "boxes": item["gt_boxes"],
            "labels": item["gt_labels"],
            "masks": item["gt_masks"],
        }
    return oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "datasets.yaml"))
    ap.add_argument("--mock", action="store_true", help="use mock predictor on synthetic data")
    ap.add_argument("--model", default=None, help="override model name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore/delete saved per-image predictions instead of resuming")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = os.path.join(REPO_ROOT, cfg.get("output_root", "results"), "predictions")
    os.makedirs(out_dir, exist_ok=True)
    if args.fresh:
        import shutil
        shutil.rmtree(os.path.join(out_dir, "per_image"), ignore_errors=True)
        print("--fresh: cleared saved per-image predictions")

    if args.mock:
        print("=== MOCK MODE: synthetic dataset + mock predictor ===")
        dataset = SyntheticDataset(n=12, size=256, seed=cfg["mock"]["seed"])
        oracle = build_oracle_gt(dataset)
        model = ModelWrapper(
            mode="mock", seed=cfg["mock"]["seed"], oracle_gt=oracle,
            mock_noise=cfg["mock"]["noise"], mock_miss_prob=cfg["mock"]["miss_prob"],
            mock_fp_prob=cfg["mock"]["fp_prob"], mock_reject_prob=cfg["mock"]["reject_prob"],
        )
        run_inference_on_dataset(model, dataset, task="detection", queries=["polyp", "cell"],
                                 out_dir=out_dir, tag="mock_synth_detection")
        run_inference_on_dataset(model, dataset, task="segmentation", queries=["polyp"],
                                 out_dir=out_dir, tag="mock_synth_segmentation")
        run_inference_on_dataset(model, dataset, task="count_box", queries=["polyp"],
                                 out_dir=out_dir, tag="mock_synth_counting")
        print("Mock inference complete. Run: python scripts/evaluate.py --mock")
        return

    # Real mode
    model_name = args.model or cfg["model"]["name"]
    device = args.device or cfg["model"]["device"]
    print(f"Loading model {model_name} on device={device} ...")
    model = ModelWrapper(mode="real", model_name=model_name, device=device,
                         dtype=cfg["model"]["dtype"],
                         max_pixels=int(cfg["model"].get("max_pixels", 768 * 32 * 32)))
    mnt = int(cfg.get("max_new_tokens", 1024))

    data_root = os.path.join(REPO_ROOT, cfg.get("data_root", "data"))

    # Kvasir-SEG
    kv = KvasirSegDataset(root=data_root, subset=cfg["kvasir"]["subset"], seed=cfg["kvasir"]["seed"])
    print(f"Kvasir-SEG: {len(kv)} images")
    for q in cfg["kvasir"]["tasks"]["referring_segmentation"]["queries"]:
        run_inference_on_dataset(model, kv, task="segmentation", queries=q,
                                 out_dir=out_dir, tag=f"kvasir_seg_{q.replace(' ', '_')}",
                                 max_new_tokens=mnt)
    run_inference_on_dataset(model, kv, task="detection",
                             queries=cfg["kvasir"]["tasks"]["detection"]["queries"][0],
                             out_dir=out_dir, tag="kvasir_detection", max_new_tokens=mnt)
    for q in cfg["kvasir"]["tasks"]["rejection"]["negative_queries"]:
        run_inference_on_dataset(model, kv, task="detection", queries=q,
                                 out_dir=out_dir, tag=f"kvasir_rejection_neg_{q.replace(' ', '_')}",
                                 max_new_tokens=mnt)

    # BCCD
    bccd = BCCDDataset(root=data_root, subset=cfg["bccd"]["subset"],
                       include_platelets=cfg["bccd"]["include_platelets"], seed=cfg["bccd"]["seed"])
    print(f"BCCD: {len(bccd)} images")
    run_inference_on_dataset(model, bccd, task="detection",
                             queries=cfg["bccd"]["tasks"]["open_vocab_detection"]["queries"],
                             out_dir=out_dir, tag="bccd_detection", max_new_tokens=mnt)
    run_inference_on_dataset(model, bccd, task="count_box",
                             queries=cfg["bccd"]["tasks"]["counting"]["query"],
                             out_dir=out_dir, tag="bccd_counting_wbc", max_new_tokens=mnt)
    print("Real inference complete. Run: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
