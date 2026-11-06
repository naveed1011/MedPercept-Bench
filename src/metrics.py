"""Metric implementations for MedPercept-Bench.

All metrics are computed per-image then aggregated. Implementations are small but
correct (no external detection-metric library). Never returns NaN; empty predictions
yield 0 with a documented convention.

Detection:
    box_iou()                 -> pairwise IoU matrix
    compute_ap_class()        -> Average Precision @ IoU>=0.5 (all-point interpolation)
    evaluate_detection()      -> mAP@0.5 + per-class P/R/F1@0.5 (micro + macro)

Segmentation:
    dice_coeff(), mask_iou()  -> binary mask overlap (pred resized to GT shape)
    evaluate_segmentation()   -> mean Dice / mean IoU over images (union of instances)

Counting:
    evaluate_counting()       -> MAE, RMSE, exact-match accuracy

Rejection:
    evaluate_rejection()      -> true-rejection rate (neg queries), false-rejection rate (pos queries)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes. Returns (N,M)."""
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate_detection(preds: list, gts: list, iou_thresh: float = 0.5) -> dict:
    """Evaluate detection over a dataset.

    preds: list (per image) of {"boxes": (N,4), "labels": list[str], "scores": list[float]}
    gts:   list (per image) of {"boxes": (M,4), "labels": list[str]}
    Returns {"mAP": float, "per_class": {cls: {ap, precision, recall, f1, tp, fp, fn}},
             "micro": {precision, recall, f1}, "iou_thresh": iou_thresh}.
    """
    classes = sorted({l for g in gts for l in g["labels"]} | {l for p in preds for l in p["labels"]})
    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    for cls in classes:
        # gather flat preds for this class with image index
        pred_scores, pred_boxes, pred_img_idx = [], [], []
        for i, p in enumerate(preds):
            for j, lab in enumerate(p["labels"]):
                if lab == cls:
                    pred_scores.append(p["scores"][j] if j < len(p["scores"]) else 1.0)
                    pred_boxes.append(np.asarray(p["boxes"]).reshape(-1, 4)[j])
                    pred_img_idx.append(i)
        gt_boxes_per_img = []
        for g in gts:
            gb = np.asarray(g["boxes"]).reshape(-1, 4)
            gl = g["labels"]
            cls_boxes = np.asarray([gb[k] for k in range(len(gl)) if gl[k] == cls], dtype=np.float64).reshape(-1, 4)
            gt_boxes_per_img.append(cls_boxes)
        n_gt = sum(len(g) for g in gt_boxes_per_img)

        if len(pred_scores) == 0:
            per_class[cls] = {"ap": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                              "tp": 0, "fp": 0, "fn": n_gt}
            micro_fn += n_gt
            continue

        order = np.argsort(-np.asarray(pred_scores, dtype=np.float64))
        gt_used = [np.zeros(len(g), dtype=bool) for g in gt_boxes_per_img]
        tp_arr = np.zeros(len(order), dtype=np.float64)
        fp_arr = np.zeros(len(order), dtype=np.float64)
        for rank, oi in enumerate(order):
            img_i = pred_img_idx[oi]
            pb = np.asarray(pred_boxes[oi], dtype=np.float64).reshape(1, 4)
            gbs = gt_boxes_per_img[img_i]
            if len(gbs) == 0:
                fp_arr[rank] = 1.0
                continue
            ious = box_iou(pb, gbs)[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_thresh and not gt_used[img_i][best]:
                tp_arr[rank] = 1.0
                gt_used[img_i][best] = True
            else:
                fp_arr[rank] = 1.0

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        recall = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
        precision = tp_cum / (tp_cum + fp_cum)
        # all-point interpolation: make precision monotonically decreasing from the end
        precision_mono = np.maximum.accumulate(precision[::-1])[::-1]
        # AP = sum over recall increments
        recall_pad = np.concatenate(([0.0], recall))
        ap = float(np.sum((recall_pad[1:] - recall_pad[:-1]) * precision_mono))

        tp_total = int(tp_arr.sum())
        fp_total = int(fp_arr.sum())
        fn_total = n_gt - tp_total
        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / n_gt if n_gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"ap": ap, "precision": prec, "recall": rec, "f1": f1,
                          "tp": tp_total, "fp": fp_total, "fn": fn_total}
        micro_tp += tp_total
        micro_fp += fp_total
        micro_fn += fn_total

    aps = [v["ap"] for v in per_class.values()]
    mAP = float(np.mean(aps)) if aps else 0.0
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    return {
        "mAP": mAP,
        "per_class": per_class,
        "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
        "iou_thresh": iou_thresh,
    }


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def _resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape[:2] == target_hw:
        return mask.astype(np.uint8)
    if _HAS_CV2:
        return cv2.resize(mask.astype(np.uint8), (target_hw[1], target_hw[0]),
                          interpolation=cv2.INTER_NEAREST)
    # PIL fallback
    from PIL import Image
    return np.array(Image.fromarray(mask.astype(np.uint8)).resize(
        (target_hw[1], target_hw[0]), Image.NEAREST))


def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        pred = _resize_mask(pred.astype(np.uint8), gt.shape).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


def mask_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        pred = _resize_mask(pred.astype(np.uint8), gt.shape).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def evaluate_segmentation(pred_masks_per_image: list, gt_masks_per_image: list) -> dict:
    """Each entry is a list of binary masks (instances). We union instances per image.

    Empty prediction -> Dice=0, IoU=0 for that image (documented convention).
    Returns {"mean_dice": float, "mean_iou": float, "per_image": [{"dice":, "iou":}]}.
    """
    per_image = []
    for preds, gts in zip(pred_masks_per_image, gt_masks_per_image):
        if gts:
            gt_union = np.logical_or.reduce([np.asarray(g).astype(bool) for g in gts])
        else:
            # no GT — if pred also empty, dice=1 (both empty); else 0
            if not preds:
                per_image.append({"dice": 1.0, "iou": 1.0})
            else:
                per_image.append({"dice": 0.0, "iou": 0.0})
            continue
        if not preds:
            per_image.append({"dice": 0.0, "iou": 0.0})
            continue
        pred_union = np.logical_or.reduce([np.asarray(p).astype(bool) for p in preds])
        per_image.append({"dice": dice_coeff(pred_union, gt_union),
                          "iou": mask_iou(pred_union, gt_union)})
    mean_dice = float(np.mean([p["dice"] for p in per_image])) if per_image else 0.0
    mean_iou = float(np.mean([p["iou"] for p in per_image])) if per_image else 0.0
    return {"mean_dice": mean_dice, "mean_iou": mean_iou, "per_image": per_image}


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def evaluate_counting(pred_counts: list, gt_counts: list) -> dict:
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    gt_counts = np.asarray(gt_counts, dtype=np.float64)
    if len(pred_counts) == 0:
        return {"mae": 0.0, "rmse": 0.0, "exact_acc": 0.0, "n": 0}
    err = pred_counts - gt_counts
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    exact = float(np.mean(err == 0))
    return {"mae": mae, "rmse": rmse, "exact_acc": exact, "n": int(len(pred_counts))}


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------
def evaluate_rejection(negative_rejected: list, positive_rejected: list) -> dict:
    """negative_rejected: list[bool] — True if model correctly returned None on negative query.
    positive_rejected: list[bool] — True if model incorrectly returned None when target present.
    """
    trr = float(np.mean(negative_rejected)) if negative_rejected else 0.0
    frr = float(np.mean(positive_rejected)) if positive_rejected else 0.0
    return {
        "true_rejection_rate": trr,    # higher is better (on negative queries)
        "false_rejection_rate": frr,   # lower is better (on positive queries)
        "n_negative": len(negative_rejected),
        "n_positive": len(positive_rejected),
    }
