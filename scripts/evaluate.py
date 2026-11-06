#!/usr/bin/env python3
"""Evaluate predictions against GT and write metrics summary.

Usage:
    python scripts/evaluate.py            # read results/predictions/*.json + real GT
    python scripts/evaluate.py --mock     # end-to-end on synthetic data with mock predictor

Outputs:
    results/metrics_summary.json
    results/metrics_summary.csv
    results/figures/metrics_bar_chart.png
    (printed) markdown summary table
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.metrics import (  # noqa: E402
    evaluate_detection, evaluate_segmentation, evaluate_counting, evaluate_rejection,
)
from src.visualize import save_side_by_side, save_metrics_bar_chart  # noqa: E402
from src.datasets import SyntheticDataset, KvasirSegDataset, BCCDDataset  # noqa: E402
from src.model_wrapper import ModelWrapper  # noqa: E402


def _b64_to_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(raw)).convert("L")) > 127


def _load_predictions(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _ok(predictions: dict) -> dict:
    """Drop images where inference failed (e.g. GPU OOM) so they are not scored as 'no detection'."""
    return {k: v for k, v in predictions.items() if not v.get("error")}


def _gt_lookup(dataset, keep_images: bool = False) -> dict:
    # Keep annotations only. Holding every decoded image in RAM was a second source of
    # out-of-memory crashes (Kvasir images are up to 1920x1072) - evaluation never needs pixels.
    out = {}
    for i in range(len(dataset)):
        item = dataset[i]
        out[item["image_id"]] = item if keep_images else {k: v for k, v in item.items() if k != "image"}
    return out


def _eval_detection_from_preds(predictions: dict, gt_lookup: dict, iou_thresh: float,
                               allowed_labels=None) -> dict:
    preds, gts = [], []
    for image_id, rec in _ok(predictions).items():
        gt = gt_lookup.get(image_id)
        if gt is None:
            continue
        preds.append({"boxes": rec["boxes_xyxy"], "labels": rec["labels"], "scores": rec["scores"]})
        gb = gt["gt_boxes"]; gl = gt["gt_labels"]
        if allowed_labels:
            keep = [i for i, l in enumerate(gl) if l in allowed_labels]
            gb = [gb[i] for i in keep] if len(gb) else []
            gl = [gl[i] for i in keep]
        gts.append({"boxes": gb, "labels": gl})
    if not preds:
        return {"mAP": 0.0, "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "per_class": {}}
    return evaluate_detection(preds, gts, iou_thresh=iou_thresh)


def _eval_segmentation_from_preds(predictions: dict, gt_lookup: dict) -> dict:
    pred_masks, gt_masks = [], []
    for image_id, rec in _ok(predictions).items():
        gt = gt_lookup.get(image_id)
        if gt is None:
            continue
        pm = [_b64_to_mask(m) for m in rec.get("masks_b64_png", [])]
        if gt["gt_masks"]:  # defensive: make predicted masks match the GT resolution
            th, tw = np.asarray(gt["gt_masks"][0]).shape[:2]
            pm = [m if m.shape == (th, tw) else
                  np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((tw, th))) > 127
                  for m in pm]
        pred_masks.append(pm)
        gt_masks.append(gt["gt_masks"])
    if not pred_masks:
        return {"mean_dice": 0.0, "mean_iou": 0.0, "per_image": []}
    return evaluate_segmentation(pred_masks, gt_masks)


def _eval_counting_from_preds(predictions: dict, gt_lookup: dict, count_label: str = None) -> dict:
    pred_counts, gt_counts = [], []
    for image_id, rec in _ok(predictions).items():
        gt = gt_lookup.get(image_id)
        if gt is None:
            continue
        pred_counts.append(rec["count"])
        if count_label:
            gt_counts.append(sum(1 for l in gt["gt_labels"] if l == count_label))
        else:
            gt_counts.append(gt["gt_count"])
    return evaluate_counting(pred_counts, gt_counts)


def _print_markdown(summary: dict, title: str):
    print(f"\n## {title}\n")
    print("| Metric | Value |")
    print("|---|---|")
    rows = _flatten_metrics(summary)
    for k, v in rows:
        print(f"| {k} | {v} |")
    print()


def _flatten_metrics(summary: dict) -> list:
    rows = []
    for section, vals in summary.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    rows.append((f"{section}/{k}", round(float(v), 4)))
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, np.floating, np.integer)):
                            rows.append((f"{section}/{k}/{kk}", round(float(vv), 4)))
        elif isinstance(vals, (int, float, np.floating, np.integer)):
            rows.append((section, round(float(vals), 4)))
    return rows


def run_mock_pipeline(cfg: dict) -> dict:
    """End-to-end mock pipeline: synthetic data -> mock predictor -> metrics -> outputs."""
    print("=" * 60)
    print("MOCK PIPELINE (synthetic data, mock predictor)")
    print("=" * 60)
    dataset = SyntheticDataset(n=12, size=256, seed=cfg["mock"]["seed"])
    gt_lookup = _gt_lookup(dataset, keep_images=True)
    oracle = {iid: {"boxes": it["gt_boxes"], "labels": it["gt_labels"], "masks": it["gt_masks"]}
              for iid, it in gt_lookup.items()}
    model = ModelWrapper(
        mode="mock", seed=cfg["mock"]["seed"], oracle_gt=oracle,
        mock_noise=cfg["mock"]["noise"], mock_miss_prob=cfg["mock"]["miss_prob"],
        mock_fp_prob=cfg["mock"]["fp_prob"], mock_reject_prob=cfg["mock"]["reject_prob"],
    )

    # 1. Detection (queries: polyp, cell)
    det_preds = {}
    for iid, it in gt_lookup.items():
        pred = model.predict(it["image"], task="detection", query=["polyp", "cell"], image_id=iid)
        det_preds[iid] = {"boxes_xyxy": pred["boxes_xyxy"].tolist(), "labels": pred["labels"],
                          "scores": pred["scores"], "count": pred["count"], "rejected": pred["rejected"]}
    det_metrics = _eval_detection_from_preds(det_preds, gt_lookup, cfg["iou_threshold"])

    # 2. Segmentation (query: polyp)
    seg_preds = {}
    for iid, it in gt_lookup.items():
        pred = model.predict(it["image"], task="segmentation", query="polyp", image_id=iid)
        import base64 as _b64, io as _io
        masks_b64 = []
        for m in pred["masks"]:
            buf = _io.BytesIO()
            Image.fromarray((np.asarray(m) > 0).astype(np.uint8) * 255).save(buf, format="PNG")
            masks_b64.append(_b64.b64encode(buf.getvalue()).decode())
        seg_preds[iid] = {"masks_b64_png": masks_b64, "count": pred["count"], "rejected": pred["rejected"]}
    seg_metrics = _eval_segmentation_from_preds(seg_preds, gt_lookup)

    # 3. Counting (count_box query: polyp)
    count_preds = {}
    for iid, it in gt_lookup.items():
        pred = model.predict(it["image"], task="count_box", query="polyp", image_id=iid)
        count_preds[iid] = {"count": pred["count"]}
    count_metrics = _eval_counting_from_preds(count_preds, gt_lookup, count_label="polyp")

    # 4. Rejection: negative query "giraffe" (target absent) vs positive "polyp" (target present)
    neg_rejected = []
    for iid, it in gt_lookup.items():
        pred = model.predict(it["image"], task="detection", query="giraffe", image_id=iid)
        neg_rejected.append(pred["rejected"])
    pos_rejected = [det_preds[iid]["rejected"] for iid in gt_lookup]  # positive query = polyp/cell
    rej_metrics = evaluate_rejection(neg_rejected, pos_rejected)

    summary = {
        "mock_detection": {"mAP@0.5": det_metrics["mAP"],
                           "micro_precision": det_metrics["micro"]["precision"],
                           "micro_recall": det_metrics["micro"]["recall"],
                           "micro_f1": det_metrics["micro"]["f1"]},
        "mock_segmentation": {"mean_dice": seg_metrics["mean_dice"], "mean_iou": seg_metrics["mean_iou"]},
        "mock_counting": {"MAE": count_metrics["mae"], "RMSE": count_metrics["rmse"],
                          "exact_acc": count_metrics["exact_acc"]},
        "mock_rejection": {"true_rejection_rate": rej_metrics["true_rejection_rate"],
                           "false_rejection_rate": rej_metrics["false_rejection_rate"]},
        "note": "MOCK predictor on synthetic data — these numbers validate the pipeline, NOT model performance.",
    }
    return summary, det_preds, seg_preds, dataset


def run_real_pipeline(cfg: dict) -> dict:
    """Read real predictions from disk and evaluate against GT datasets."""
    data_root = os.path.join(REPO_ROOT, cfg.get("data_root", "data"))
    pred_dir = os.path.join(REPO_ROOT, cfg.get("output_root", "results"), "predictions")
    iou = cfg["iou_threshold"]
    summary = {}

    kv = KvasirSegDataset(root=data_root, subset=cfg["kvasir"]["subset"], seed=cfg["kvasir"]["seed"])
    kv_gt = _gt_lookup(kv) if len(kv) else {}
    bccd = BCCDDataset(root=data_root, subset=cfg["bccd"]["subset"],
                       include_platelets=cfg["bccd"]["include_platelets"], seed=cfg["bccd"]["seed"])
    bccd_gt = _gt_lookup(bccd) if len(bccd) else {}

    # Kvasir detection
    p = _load_predictions(os.path.join(pred_dir, "kvasir_detection.json"))
    if p:
        m = _eval_detection_from_preds(p, kv_gt, iou)
        summary["kvasir_detection"] = {"mAP@0.5": m["mAP"], "micro_f1": m["micro"]["f1"],
                                       "n_images": len(_ok(p)), "n_oom_skipped": len(p) - len(_ok(p))}
    # Kvasir segmentation (referring, both queries)
    for q in cfg["kvasir"]["tasks"]["referring_segmentation"]["queries"]:
        tag = f"kvasir_seg_{q.replace(' ', '_')}"
        p = _load_predictions(os.path.join(pred_dir, f"{tag}.json"))
        if p:
            m = _eval_segmentation_from_preds(p, kv_gt)
            summary[tag] = {"mean_dice": m["mean_dice"], "mean_iou": m["mean_iou"], "n_images": len(_ok(p)), "n_oom_skipped": len(p) - len(_ok(p))}
    # Kvasir rejection
    neg_flags, pos_flags = [], []
    for q in cfg["kvasir"]["tasks"]["rejection"]["negative_queries"]:
        tag = f"kvasir_rejection_neg_{q.replace(' ', '_')}"
        p = _load_predictions(os.path.join(pred_dir, f"{tag}.json"))
        neg_flags.extend([rec["rejected"] for rec in _ok(p).values()])
    ppos = _load_predictions(os.path.join(pred_dir, "kvasir_detection.json"))
    pos_flags = [rec["rejected"] for rec in _ok(ppos).values()]
    if neg_flags or pos_flags:
        r = evaluate_rejection(neg_flags, pos_flags)
        summary["kvasir_rejection"] = {"true_rejection_rate": r["true_rejection_rate"],
                                       "false_rejection_rate": r["false_rejection_rate"],
                                       "n_neg": r["n_negative"], "n_pos": r["n_positive"]}
    # BCCD detection
    p = _load_predictions(os.path.join(pred_dir, "bccd_detection.json"))
    if p:
        m = _eval_detection_from_preds(p, bccd_gt, iou,
                                       allowed_labels={"white blood cell", "red blood cell"})
        summary["bccd_detection"] = {"mAP@0.5": m["mAP"], "micro_f1": m["micro"]["f1"],
                                     "per_class_ap": {k: v["ap"] for k, v in m["per_class"].items()},
                                     "n_images": len(_ok(p)), "n_oom_skipped": len(p) - len(_ok(p))}
    # BCCD counting (WBC)
    p = _load_predictions(os.path.join(pred_dir, "bccd_counting_wbc.json"))
    if p:
        m = _eval_counting_from_preds(p, bccd_gt, count_label="white blood cell")
        summary["bccd_counting_wbc"] = {"MAE": m["mae"], "RMSE": m["rmse"], "exact_acc": m["exact_acc"],
                                        "n_images": m["n"]}
    if not summary:
        summary["note"] = "No real predictions found. Run scripts/run_inference.py first (or use --mock)."
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "datasets.yaml"))
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = os.path.join(REPO_ROOT, cfg.get("output_root", "results"))
    fig_dir = os.path.join(out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if args.mock:
        summary, det_preds, seg_preds, dataset = run_mock_pipeline(cfg)
        title = "MedPercept-Bench MOCK pipeline metrics (synthetic data)"
        # save a couple of side-by-side visualizations
        gt_lookup = _gt_lookup(dataset, keep_images=True)
        for iid in list(gt_lookup.keys())[:3]:
            it = gt_lookup[iid]
            save_side_by_side(
                it["image"], it["gt_boxes"], it["gt_labels"], it["gt_masks"],
                np.asarray(det_preds[iid]["boxes_xyxy"]), det_preds[iid]["labels"],
                [_b64_to_mask(m) for m in seg_preds[iid]["masks_b64_png"]],
                os.path.join(fig_dir, f"mock_overlay_{iid}.png"), title=iid,
            )
    else:
        summary = run_real_pipeline(cfg)
        title = "MedPercept-Bench metrics (real predictions)"

    # Write JSON
    json_path = os.path.join(out_root, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    # Write CSV
    rows = _flatten_metrics(summary)
    df = pd.DataFrame(rows, columns=["metric", "value"])
    csv_path = os.path.join(out_root, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    # Bar chart (only numeric scalar metrics, cap to first 12)
    chart_vals = {k: v for k, v in rows if isinstance(v, (int, float))}
    if chart_vals:
        keys = list(chart_vals.keys())[:12]
        save_metrics_bar_chart({k: chart_vals[k] for k in keys},
                               os.path.join(fig_dir, "metrics_bar_chart.png"), title=title)
    # Print markdown
    _print_markdown(summary, title)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Figure: {os.path.join(fig_dir, 'metrics_bar_chart.png')}")


if __name__ == "__main__":
    main()
