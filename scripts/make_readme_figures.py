#!/usr/bin/env python3
"""Build the README figures from the REAL results in results/ (no numbers are typed in by hand).

Usage: python scripts/make_readme_figures.py
Writes assets/figures/*.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "assets", "figures")
os.makedirs(OUT, exist_ok=True)

# Gallery examples shown in the README (picked by eye from results/figures; all 13 are kept there).
SUCCESS = ["cju1erep75us208553i4ofwwe", "cju1dq3x1vgx109889c7wyirg", "cju2pmhtr17a00855cvpelzb0"]
FAILURE = ["cju1dia8hvc6v098827mgffnm", "cju1hyolc7aqu0878rrkfn1lr"]


def stack(ids, name, width=1300):
    ims = []
    for i in ids:
        p = os.path.join(RES, "figures", f"gallery_kvasir_{i}.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            ims.append(im.resize((width, int(im.height * width / im.width))))
    if not ims:
        return
    sheet = Image.new("RGB", (width, sum(i.height for i in ims)), "white")
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(os.path.join(OUT, name), optimize=True)


def overview():
    m = json.load(open(os.path.join(RES, "metrics_summary.json")))
    pred = os.path.join(RES, "predictions")
    rej = {}
    for q in ["giraffe", "white_blood_cell"]:
        p = os.path.join(pred, f"kvasir_rejection_neg_{q}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            rej[q.replace("_", " ")] = sum(v["rejected"] for v in d.values()) / max(len(d), 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    kv = [("Detection\nmAP@0.5", m["kvasir_detection"]["mAP@0.5"]),
          ("Detection\nF1", m["kvasir_detection"]["micro_f1"]),
          ("Segmentation\nDice", m["kvasir_seg_polyp"]["mean_dice"]),
          ("Segmentation\nIoU", m["kvasir_seg_polyp"]["mean_iou"])]
    b = ax[0].bar([k for k, _ in kv], [v for _, v in kv], color="#2a9d8f")
    ax[0].set_ylim(0, 1); ax[0].set_title(f"Kvasir-SEG polyps (n={m['kvasir_seg_polyp']['n_images']})")
    ax[0].bar_label(b, fmt="%.2f")

    items = list(rej.items())
    if items:
        b = ax[1].bar([f"'{k}'\nabsent query" for k, _ in items], [v for _, v in items], color="#e9c46a")
        ax[1].bar_label(b, fmt="%.2f")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Rejection: how often it correctly says 'None'\n(higher is better; positives wrongly rejected: "
                    f"{m['kvasir_rejection']['false_rejection_rate']:.0%})", fontsize=10)

    bc = [("RBC\nAP@0.5", m["bccd_detection"]["per_class_ap"]["red blood cell"]),
          ("WBC\nAP@0.5", m["bccd_detection"]["per_class_ap"]["white blood cell"])]
    b = ax[2].bar([k for k, _ in bc], [v for _, v in bc], color="#e76f51")
    ax[2].set_ylim(0, 1); ax[2].bar_label(b, fmt="%.2f")
    ax[2].set_title(f"BCCD blood cells (n={m['bccd_detection']['n_images']})\nWBC count MAE = {m['bccd_counting_wbc']['MAE']:.1f}")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "results_overview.png"), dpi=150)


if __name__ == "__main__":
    stack(SUCCESS, "kvasir_good_examples.png")
    stack(FAILURE, "kvasir_failure_cases.png")
    overview()
    print("Wrote figures to", OUT)
