"""Unit tests for src/metrics.py using synthetic, hand-computed cases.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    box_iou, dice_coeff, mask_iou,
    evaluate_detection, evaluate_segmentation, evaluate_counting, evaluate_rejection,
)


# ---------------------------------------------------------------------------
# Box IoU
# ---------------------------------------------------------------------------
def test_box_iou_identical():
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert abs(box_iou(b, b)[0, 0] - 1.0) < 1e-9


def test_box_iou_disjoint():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    assert box_iou(a, b)[0, 0] == 0.0


def test_box_iou_half_overlap():
    # a=[0,0,10,10] area=100; b=[5,0,15,10] area=100; inter=[5,0,10,10] area=50; union=150
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    assert abs(box_iou(a, b)[0, 0] - 50.0 / 150.0) < 1e-9


def test_box_iou_empty():
    a = np.zeros((0, 4))
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(a, b).shape == (0, 1)


# ---------------------------------------------------------------------------
# Dice / IoU masks
# ---------------------------------------------------------------------------
def test_dice_identical():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:8, 2:8] = 1
    assert abs(dice_coeff(m, m) - 1.0) < 1e-6


def test_dice_empty_pred():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[2:8, 2:8] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    assert dice_coeff(pred, gt) < 1e-3


def test_mask_iou_half():
    a = np.zeros((10, 10), dtype=np.uint8); a[:, :5] = 1
    b = np.zeros((10, 10), dtype=np.uint8); b[:, 5:] = 1
    # disjoint -> iou ~0
    assert mask_iou(a, b) < 1e-3


# ---------------------------------------------------------------------------
# Detection AP / mAP — hand-computed case
# ---------------------------------------------------------------------------
def test_detection_ap_hand_computed():
    # 2 images, 1 class "A". n_gt = 2.
    # img1: GT [0,0,10,10]. preds: [0,0,10,10] score .9 (TP), [20,20,30,30] score .5 (FP)
    # img2: GT [0,0,10,10]. pred: [1,1,11,11] score .8 -> IoU=81/119=0.68>0.5 (TP)
    # Sorted: .9 TP, .8 TP, .5 FP. recall=[.5,1,1], precision=[1,1,2/3]. AP=1.0.
    preds = [
        {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "labels": ["A", "A"], "scores": [0.9, 0.5]},
        {"boxes": [[1, 1, 11, 11]], "labels": ["A"], "scores": [0.8]},
    ]
    gts = [
        {"boxes": [[0, 0, 10, 10]], "labels": ["A"]},
        {"boxes": [[0, 0, 10, 10]], "labels": ["A"]},
    ]
    res = evaluate_detection(preds, gts, iou_thresh=0.5)
    assert abs(res["mAP"] - 1.0) < 1e-6
    # F1: TP=2, FP=1, FN=0 -> precision=2/3, recall=1, F1=0.8
    assert abs(res["micro"]["precision"] - 2.0 / 3.0) < 1e-6
    assert abs(res["micro"]["recall"] - 1.0) < 1e-6
    assert abs(res["micro"]["f1"] - 0.8) < 1e-6


def test_detection_no_preds():
    preds = [{"boxes": [], "labels": [], "scores": []}]
    gts = [{"boxes": [[0, 0, 10, 10]], "labels": ["A"]}]
    res = evaluate_detection(preds, gts, iou_thresh=0.5)
    assert res["mAP"] == 0.0
    assert res["micro"]["f1"] == 0.0
    assert res["per_class"]["A"]["fn"] == 1


def test_detection_two_classes():
    preds = [{"boxes": [[0, 0, 10, 10]], "labels": ["A"], "scores": [0.9]}]
    gts = [{"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "labels": ["A", "B"]}]
    res = evaluate_detection(preds, gts, iou_thresh=0.5)
    # class A AP=1, class B AP=0 -> mAP=0.5
    assert abs(res["mAP"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def test_counting_mae():
    res = evaluate_counting([1, 2, 3], [1, 1, 3])
    assert abs(res["mae"] - 1.0 / 3.0) < 1e-9
    assert abs(res["rmse"] - np.sqrt(1.0 / 3.0)) < 1e-9
    assert abs(res["exact_acc"] - 2.0 / 3.0) < 1e-9


def test_counting_empty():
    res = evaluate_counting([], [])
    assert res["n"] == 0


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------
def test_rejection():
    neg = [True, True, False, True]   # 3/4 correctly rejected
    pos = [False, False, True, False]  # 1/4 falsely rejected
    res = evaluate_rejection(neg, pos)
    assert abs(res["true_rejection_rate"] - 0.75) < 1e-9
    assert abs(res["false_rejection_rate"] - 0.25) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
