"""Dataset loaders for MedPercept-Bench.

Unified interface: every dataset __getitem__ returns a dict:
    {
        "image": PIL.Image,
        "gt_boxes": (N,4) np.ndarray float32 in PIXEL xyxy,
        "gt_labels": list[str],
        "gt_masks": list[np.ndarray] uint8 {0,1} at image HW (or [] if no masks),
        "gt_count": int,                 # for counting tasks
        "image_id": str,
        "width": int, "height": int,
    }

Datasets:
    KvasirSegDataset  — polyp segmentation (1000 imgs+masks); boxes derived from masks.
    BCCDDataset       — blood cell detection (Pascal VOC XML: WBC/RBC/Platelets).
    SyntheticDataset  — tiny synthetic data for mock/CI pipeline (no download needed).
"""

from __future__ import annotations

import os
import glob
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mask_to_boxes(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Derive per-instance boxes + binary masks from a (H,W) label mask.

    Uses connected components. Returns (boxes Nx4 pixel xyxy, masks list[HxW uint8]).
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 127).astype(np.uint8)
    if _HAS_CV2:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        masks = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 10:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            masks.append((labels == i).astype(np.uint8))
    else:  # fallback: single bounding rect of whole mask
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return np.zeros((0, 4), dtype=np.float32), []
        boxes = [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]
        masks = [binary]
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), masks


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Kvasir-SEG
# ---------------------------------------------------------------------------
class KvasirSegDataset:
    """Kvasir-SEG polyp segmentation dataset.

    Expected layout: root/Kvasir-SEG/images/*.jpg and root/Kvasir-SEG/masks/*.png.
    """

    def __init__(self, root: str = "data", subset: Optional[int] = None, seed: int = 42):
        self.root = root
        img_dir = os.path.join(root, "Kvasir-SEG", "images")
        mask_dir = os.path.join(root, "Kvasir-SEG", "masks")
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                                  glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_dir = mask_dir
        if subset is not None and subset < len(self.image_paths):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.image_paths), size=subset, replace=False)
            self.image_paths = [self.image_paths[i] for i in sorted(idx)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        image = _load_image(img_path)
        w, h = image.size
        mask_path = os.path.join(self.mask_dir, image_id + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, image_id + ".jpg")
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
        gt_masks: list[np.ndarray] = []
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            gt_boxes, gt_masks = _mask_to_boxes(mask)
        gt_labels = ["polyp"] * len(gt_boxes)
        return {
            "image": image,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_masks": gt_masks,
            "gt_count": len(gt_boxes),
            "image_id": image_id,
            "width": w,
            "height": h,
        }


# ---------------------------------------------------------------------------
# BCCD (Shenggan/BCCD_Dataset) — Pascal VOC XML
# ---------------------------------------------------------------------------
_BCCD_LABEL_MAP = {"WBC": "white blood cell", "RBC": "red blood cell", "Platelets": "platelet"}


class BCCDDataset:
    """BCCD blood-cell detection dataset.

    Expected layout: root/BCCD_Dataset-master/BCCD/JPEGImages/*.jpg
                     root/BCCD_Dataset-master/BCCD/Annotations/*.xml
    """

    def __init__(self, root: str = "data", subset: Optional[int] = None,
                 include_platelets: bool = False, seed: int = 42):
        self.root = root
        self.include_platelets = include_platelets
        base = os.path.join(root, "BCCD_Dataset-master", "BCCD")
        self.img_dir = os.path.join(base, "JPEGImages")
        self.ann_dir = os.path.join(base, "Annotations")
        self.xml_paths = sorted(glob.glob(os.path.join(self.ann_dir, "*.xml")))
        if subset is not None and subset < len(self.xml_paths):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.xml_paths), size=subset, replace=False)
            self.xml_paths = [self.xml_paths[i] for i in sorted(idx)]

    def __len__(self):
        return len(self.xml_paths)

    @staticmethod
    def _parse_voc_xml(xml_path: str) -> tuple[list[list[float]], list[str], str, int, int]:
        tree = ET.parse(xml_path)
        r = tree.getroot()
        filename = r.findtext("filename", "")
        size = r.find("size")
        width = int(size.findtext("width", "0")) if size is not None else 0
        height = int(size.findtext("height", "0")) if size is not None else 0
        boxes, labels = [], []
        for obj in r.findall("object"):
            name = obj.findtext("name", "")
            lab = _BCCD_LABEL_MAP.get(name, name)
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(lab)
        return boxes, labels, filename, width, height

    def __getitem__(self, idx):
        xml_path = self.xml_paths[idx]
        boxes, labels, filename, w, h = self._parse_voc_xml(xml_path)
        image_id = os.path.splitext(os.path.basename(xml_path))[0]
        # filter platelets if requested
        if not self.include_platelets:
            boxes = [b for b, l in zip(boxes, labels) if l != "platelet"]
            labels = [l for l in labels if l != "platelet"]
        img_path = os.path.join(self.img_dir, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, image_id + ".jpg")
        image = _load_image(img_path) if os.path.exists(img_path) else Image.new("RGB", (w or 640, h or 480))
        w, h = image.size
        gt_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32)
        gt_count = sum(1 for l in labels if l == "white blood cell")
        return {
            "image": image,
            "gt_boxes": gt_boxes,
            "gt_labels": labels,
            "gt_masks": [],
            "gt_count": gt_count,
            "image_id": image_id,
            "width": w,
            "height": h,
        }


# ---------------------------------------------------------------------------
# Synthetic dataset (for mock/CI — no download required)
# ---------------------------------------------------------------------------
class SyntheticDataset:
    """Generates tiny synthetic images with boxes + masks. Used by --mock pipeline.

    Each image has 1-3 objects of class 'polyp' (with masks) and 'cell' (boxes only),
    so detection, segmentation, counting, and rejection metrics are all exercised.
    """

    def __init__(self, n: int = 12, size: int = 256, seed: int = 42):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self._items = [self._gen(i) for i in range(n)]

    def _gen(self, i: int) -> dict:
        s = self.size
        rng = np.random.default_rng(1000 + i)
        image = (rng.integers(80, 180, size=(s, s, 3))).astype(np.uint8)
        n_polyps = int(rng.integers(1, 3))
        n_cells = int(rng.integers(0, 3))
        boxes = []
        labels = []
        masks = []
        for _ in range(n_polyps):
            cx, cy = int(rng.integers(40, s - 40)), int(rng.integers(40, s - 40))
            r = int(rng.integers(15, 40))
            yy, xx = np.ogrid[:s, :s]
            m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.uint8)
            # tint image inside mask
            image[m > 0] = (image[m > 0] * 0.5 + np.array([180, 80, 80]) * 0.5).astype(np.uint8)
            ys, xs = np.where(m > 0)
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            labels.append("polyp")
            masks.append(m)
        for _ in range(n_cells):
            cx, cy = int(rng.integers(30, s - 30)), int(rng.integers(30, s - 30))
            r = int(rng.integers(10, 25))
            boxes.append([float(cx - r), float(cy - r), float(cx + r), float(cy + r)])
            labels.append("cell")
        gt_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32)
        return {
            "image": Image.fromarray(image),
            "gt_boxes": gt_boxes,
            "gt_labels": labels,
            "gt_masks": masks,
            "gt_count": n_polyps,
            "image_id": f"synth_{i:03d}",
            "width": s,
            "height": s,
        }

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self._items[idx]
