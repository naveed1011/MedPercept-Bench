"""Model wrapper for WeVisOne-VL (real) and a deterministic mock (for CI / no-GPU).

The real backend follows the verified WeVisOne-VL API:
    from wevisone_vl import WeVisOneVL
    model = WeVisOneVL("tencent/WeVisOne-VL-2B", backend="transformers", device="auto")
    result = model.predict(image, task=<task>, query=<str|list>, max_new_tokens=4096, temperature=0.0)

`result` is an `InferenceResult` dataclass (wevisone_vl/engine.py) with attributes:
    .response: str   (raw text containing <ref>label</ref><box>x1 y1 x2 y2</box>, coords 0..1000; <OBJ> for seg)
    .prompt: str
    .boxes: list[list[float]]   normalized [0,1] xyxy, populated ONLY when <OBJ> appears (seg/matting)
    .masks: list[np.ndarray]    float [0,1] per-instance masks at original HW, ONLY for seg/matting
    .latency_s: float

Canonical parser: wevisone_vl.visualization.parse_predictions(response, width, height) ->
    {label: [{"type": "box"|"point"|"polygon", "coords": pixel coords}]}

We try to import the canonical parser; if wevisone_vl is not installed (e.g. mock/CI env),
we fall back to a local re-implementation that parses the same token grammar. This fallback
exists ONLY so the mock pipeline and unit tests can run without installing wevisone_vl.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Try to import canonical parser from wevisone_vl; fall back to local parser.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when wevisone_vl is installed
    from wevisone_vl.visualization import parse_predictions as _canonical_parse_predictions

    _HAS_WEVISONE_PARSER = True
except Exception:  # ImportError or any error inside the package
    _canonical_parse_predictions = None
    _HAS_WEVISONE_PARSER = False


# ---------------------------------------------------------------------------
# Local fallback parser (mirrors the documented WeVisOne-VL token grammar).
# ---------------------------------------------------------------------------
_BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)
_POINT_RE = re.compile(r"<point>(.*?)</point>", re.DOTALL)
_SEGM_RE = re.compile(r"<segm>(.*?)</segm>", re.DOTALL)
_REF_RE = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _local_parse_predictions(response: str, width: int, height: int) -> dict:
    """Parse <ref>..</ref><box>..</box> / <point>..</point> / <segm>..</segm> tokens.

    Coordinates inside tokens are in 0..1000 (WeVisOne-VL convention) and are
    rescaled to pixel space using (width, height). A <ref> label applies to the
    detection token that immediately follows it; detections without a preceding
    <ref> inherit the most recent label or "object".
    """
    out: dict[str, list[dict]] = {}
    # Tokenize the response in order: walk through <ref> and detection tokens.
    token_re = re.compile(r"(<ref>.*?</ref>|<box>.*?</box>|<point>.*?</point>|<segm>.*?</segm>)", re.DOTALL)
    current_label = "object"
    for tok in token_re.finditer(response):
        token = tok.group(1)
        ref_m = _REF_RE.match(token)
        if ref_m:
            current_label = ref_m.group(1).strip()
            continue
        box_m = _BOX_RE.match(token)
        if box_m:
            nums = [float(x) for x in _NUM_RE.findall(box_m.group(1))]
            if len(nums) >= 4:
                x1, y1, x2, y2 = nums[:4]
                coords = [x1 / 1000.0 * width, y1 / 1000.0 * height,
                          x2 / 1000.0 * width, y2 / 1000.0 * height]
                out.setdefault(current_label, []).append({"type": "box", "coords": coords})
            continue
        pt_m = _POINT_RE.match(token)
        if pt_m:
            nums = [float(x) for x in _NUM_RE.findall(pt_m.group(1))]
            if len(nums) >= 2:
                x, y = nums[:2]
                coords = [x / 1000.0 * width, y / 1000.0 * height]
                out.setdefault(current_label, []).append({"type": "point", "coords": coords})
            continue
        seg_m = _SEGM_RE.match(token)
        if seg_m:
            nums = [float(x) for x in _NUM_RE.findall(seg_m.group(1))]
            if len(nums) >= 6 and len(nums) % 2 == 0:
                coords = [[nums[i] / 1000.0 * width, nums[i + 1] / 1000.0 * height]
                          for i in range(0, len(nums), 2)]
                out.setdefault(current_label, []).append({"type": "polygon", "coords": coords})
            continue
    return out


def parse_predictions(response: str, width: int, height: int) -> dict:
    """Canonical parser if available, else local fallback."""
    if _HAS_WEVISONE_PARSER:
        return _canonical_parse_predictions(response, width, height)
    return _local_parse_predictions(response, width, height)


# ---------------------------------------------------------------------------
# Mock InferenceResult (same shape as wevisone_vl.engine.InferenceResult).
# ---------------------------------------------------------------------------
@dataclass
class MockInferenceResult:
    response: str = ""
    prompt: str = ""
    boxes: list = field(default_factory=list)       # normalized [0,1] xyxy, seg only
    masks: list = field(default_factory=list)        # float [0,1] HW arrays, seg only
    latency_s: float = 0.0


# ---------------------------------------------------------------------------
# Unified output builder.
# ---------------------------------------------------------------------------
def _normalize_result(result: Any, image: Image.Image, task: str) -> dict:
    """Convert a real or mock InferenceResult into the unified prediction dict.

    Unified dict keys:
        boxes_xyxy : (N,4) np.ndarray float32 in PIXEL space (empty if none)
        labels     : list[str]
        scores     : list[float]  (all 1.0 — model emits no confidence scores)
        masks      : list[np.ndarray] uint8 binary {0,1} (seg only; resized to GT later)
        count      : int  (number of parsed instances; for count_box/count_point tasks)
        rejected   : bool  (True if no instances parsed)
        raw_response: str
        latency_s  : float
    """
    w, h = image.size
    boxes_xyxy: list[list[float]] = []
    labels: list[str] = []
    masks: list[np.ndarray] = []

    if task in ("segmentation", "matting"):
        # Use result.boxes (normalized [0,1]) and result.masks directly.
        for i, nb in enumerate(result.boxes or []):
            if len(nb) < 4:
                continue
            x1, y1, x2, y2 = nb[:4]
            boxes_xyxy.append([x1 * w, y1 * h, x2 * w, y2 * h])
            labels.append("segment")
            if result.masks is not None and i < len(result.masks):
                m = np.asarray(result.masks[i], dtype=np.float32)
                if m.ndim == 3:  # HW or 1xHW or HxWxC
                    m = m.squeeze()
                # threshold at 0.5 to binary
                mb = (m >= 0.5).astype(np.uint8)
                masks.append(mb)
        count = len(boxes_xyxy)
    else:
        parsed = parse_predictions(result.response, w, h)
        for label, instances in parsed.items():
            for inst in instances:
                if inst["type"] == "box":
                    c = inst["coords"]
                    boxes_xyxy.append([float(c[0]), float(c[1]), float(c[2]), float(c[3])])
                    labels.append(label)
                # points / polygons are not box-shaped; count them as instances for
                # count_point tasks but do not add to boxes_xyxy.
        count = sum(len(v) for v in parsed.values())

    rejected = (len(boxes_xyxy) == 0) and (len(masks) == 0) and (count == 0)

    return {
        "boxes_xyxy": np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32),
        "labels": labels,
        "scores": [1.0] * len(boxes_xyxy),  # model emits no scores — documented
        "masks": masks,
        "count": int(count),
        "rejected": bool(rejected),
        "raw_response": result.response,
        "latency_s": float(getattr(result, "latency_s", 0.0)),
    }


# ---------------------------------------------------------------------------
# Model wrapper (real + mock).
# ---------------------------------------------------------------------------
class ModelWrapper:
    """Thin wrapper around WeVisOne-VL with a deterministic mock backend.

    mode:
        "real"  -> loads WeVisOneVL from HuggingFace (requires GPU recommended).
        "mock"  -> returns MockInferenceResult objects. If `oracle_gt` is provided
                   (dict image_id -> {"boxes": Nx4 pixel, "masks": list[np.ndarray],
                   "labels": list}), mock predictions are noisy perturbations of GT
                   with controlled miss / false-positive rates. This is ONLY to make
                   the mock pipeline exercise non-trivial metric values; it is not a
                   model simulation and must never be reported as real results.
    """

    def __init__(
        self,
        mode: str = "mock",
        model_name: str = "tencent/WeVisOne-VL-2B",
        device: str = "auto",
        dtype: str = "bfloat16",
        seed: int = 42,
        oracle_gt: Optional[dict] = None,
        mock_noise: float = 0.05,       # fraction of image size, box jitter
        mock_miss_prob: float = 0.2,    # probability of dropping a GT instance
        mock_fp_prob: float = 0.1,      # probability of adding a false positive
        mock_reject_prob: float = 0.0,  # probability of returning empty response
        max_pixels: int = 768 * 32 * 32,  # real mode: cap on image size seen by the VLM (memory knob)
    ):
        self.mode = mode
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.oracle_gt = oracle_gt or {}
        self.mock_noise = mock_noise
        self.mock_miss_prob = mock_miss_prob
        self.mock_fp_prob = mock_fp_prob
        self.mock_reject_prob = mock_reject_prob
        self.max_pixels = int(max_pixels)
        self._model = None

        if mode == "real":
            self._load_real_model()

    def _load_real_model(self):  # pragma: no cover - requires weights download
        from wevisone_vl import WeVisOneVL

        # MEMORY NOTE: the upstream default is max_pixels = 2025*32*32 (~2000 visual tokens per
        # image). That is what made T4 runs run out of memory. We cap it (default 768*32*32, about
        # 0.8 megapixels). Boxes are predicted in normalised 0..1000 coordinates, so down-scaling
        # inside the processor does not change the coordinate system we evaluate in.
        self._model = WeVisOneVL(
            self.model_name,
            backend="transformers",
            device=self.device,
            dtype=self.dtype,
            max_pixels=self.max_pixels,
        )

    # -- mock helpers -------------------------------------------------------
    def _mock_response_from_boxes(self, boxes_px: list, labels: list, w: int, h: int) -> str:
        """Build a response string with canonical WeVisOne-VL tokens.

        Exact grammar (wevisone_vl/visualization.py REF_RE + parse_predictions):
            <ref>label</ref><box><x1><y1><x2><y2></box>
        Coordinates are 0..1000, each wrapped in angle brackets. Multiple boxes
        per <ref> are comma-separated inside one <box>...</box> block.
        """
        parts = []
        for box, lab in zip(boxes_px, labels):
            x1, y1, x2, y2 = box
            cx1 = max(0, min(999, int(round(x1 / w * 1000))))
            cy1 = max(0, min(999, int(round(y1 / h * 1000))))
            cx2 = max(1, min(1000, int(round(x2 / w * 1000))))
            cy2 = max(1, min(1000, int(round(y2 / h * 1000))))
            if cx2 <= cx1:
                cx2 = min(1000, cx1 + 1)
            if cy2 <= cy1:
                cy2 = min(1000, cy1 + 1)
            parts.append(f"<ref>{lab}</ref><box><{cx1}><{cy1}><{cx2}><{cy2}></box>")
        return "".join(parts)

    def _mock_predict(self, image: Image.Image, task: str, query, image_id: Optional[str] = None) -> MockInferenceResult:
        t0 = time.time()
        w, h = image.size
        gt = self.oracle_gt.get(image_id, {}) if image_id else {}

        if self.rng.random() < self.mock_reject_prob:
            return MockInferenceResult(response="", prompt="", boxes=[], masks=[], latency_s=time.time() - t0)

        if task in ("segmentation", "matting"):
            gt_masks = gt.get("masks", [])
            out_boxes_norm = []
            out_masks = []
            for m in gt_masks:
                if self.rng.random() < self.mock_miss_prob:
                    continue
                m = np.asarray(m, dtype=np.float32)
                if m.shape[:2] != (h, w):
                    import cv2
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                # noisy mask: erode/dilate randomly
                import cv2
                k = int(self.rng.integers(1, 4))
                kernel = np.ones((k, k), np.uint8)
                if self.rng.random() < 0.5:
                    m = cv2.erode(m, kernel, iterations=1)
                else:
                    m = cv2.dilate(m, kernel, iterations=1)
                ys, xs = np.where(m > 0.5)
                if len(xs) == 0:
                    continue
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                out_boxes_norm.append([x1 / w, y1 / h, x2 / w, y2 / h])
                out_masks.append(np.clip(m, 0.0, 1.0).astype(np.float32))
            # false positive masks
            if self.rng.random() < self.mock_fp_prob:
                fw = int(w * 0.1)
                fh = int(h * 0.1)
                cx = int(self.rng.integers(fw, w - fw))
                cy = int(self.rng.integers(fh, h - fh))
                fm = np.zeros((h, w), dtype=np.float32)
                fm[cy - fh:cy + fh, cx - fw:cx + fw] = 1.0
                out_boxes_norm.append([(cx - fw) / w, (cy - fh) / h, (cx + fw) / w, (cy + fh) / h])
                out_masks.append(fm)
            response = "<OBJ>polyp</OBJ>" if out_boxes_norm else ""
            return MockInferenceResult(response=response, prompt="", boxes=out_boxes_norm, masks=out_masks, latency_s=time.time() - t0)

        # detection / referring / count_box / point / ocr etc. -> use <ref>/<box> tokens
        gt_boxes = gt.get("boxes", np.zeros((0, 4)))
        gt_labels = gt.get("labels", [])
        pred_boxes = []
        pred_labels = []
        for i in range(len(gt_boxes)):
            if self.rng.random() < self.mock_miss_prob:
                continue
            x1, y1, x2, y2 = gt_boxes[i]
            bw, bh = x2 - x1, y2 - y1
            jx = self.rng.normal(0, self.mock_noise) * w
            jy = self.rng.normal(0, self.mock_noise) * h
            jw = self.rng.normal(0, self.mock_noise) * w
            jh = self.rng.normal(0, self.mock_noise) * h
            pred_boxes.append([max(0, x1 + jx), max(0, y1 + jy), min(w, x2 + jw), min(h, y2 + jh)])
            lab = gt_labels[i] if i < len(gt_labels) else "object"
            pred_labels.append(lab)
        # false positives
        if self.rng.random() < self.mock_fp_prob:
            bw = int(w * 0.1)
            bh = int(h * 0.1)
            cx = int(self.rng.integers(bw, w - bw))
            cy = int(self.rng.integers(bh, h - bh))
            pred_boxes.append([cx - bw, cy - bh, cx + bw, cy + bh])
            pred_labels.append(gt_labels[0] if gt_labels else "object")

        # For rejection (negative query) the mock should usually reject: if the
        # query label is not present in gt_labels, drop all preds with high prob.
        qstr = query if isinstance(query, str) else " ".join(query)
        if gt_labels and not any(l.lower() in qstr.lower() for l in gt_labels):
            # negative query: reject most of the time
            if self.rng.random() < 0.8:
                pred_boxes, pred_labels = [], []

        response = self._mock_response_from_boxes(pred_boxes, pred_labels, w, h)
        return MockInferenceResult(response=response, prompt="", boxes=[], masks=[], latency_s=time.time() - t0)

    # -- public API ---------------------------------------------------------
    @staticmethod
    def _free_gpu():  # pragma: no cover - GPU only
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def predict(self, image: Image.Image, task: str, query: Union[str, list],
                image_id: Optional[str] = None, max_new_tokens: int = 1024,
                temperature: float = 0.0) -> dict:
        """Run one request. Never raises on GPU OOM: returns a record with ``error="oom"``.

        An OOM on one hard image should not kill a 1-hour Colab run, so we free the cache, mark
        the image as errored (it is reported separately, never silently counted as a rejection),
        and carry on.
        """
        if self.mode == "mock":
            result = self._mock_predict(image, task, query, image_id=image_id)
            return _normalize_result(result, image, task)

        try:  # pragma: no cover - requires real model
            result = self._model.predict(image, task=task, query=query,
                                         max_new_tokens=max_new_tokens, temperature=temperature)
            out = _normalize_result(result, image, task)
            del result
            return out
        except RuntimeError as exc:  # pragma: no cover
            if "out of memory" not in str(exc).lower():
                raise
            self._free_gpu()
            return {
                "boxes_xyxy": np.zeros((0, 4), dtype=np.float32), "labels": [], "scores": [],
                "masks": [], "count": 0, "rejected": False, "raw_response": "",
                "latency_s": 0.0, "error": "oom",
            }

    def predict_batch(self, images, task: str, query: Union[str, list],
                      image_ids: Optional[list] = None, **kwargs) -> list[dict]:
        out = []
        for i, img in enumerate(images):
            iid = image_ids[i] if image_ids else None
            out.append(self.predict(img, task, query, image_id=iid, **kwargs))
        return out
