"""Evaluate fixed predictions against original and converted ground truth."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from .ap import APTable, bootstrap_weights, run_cocoeval

MEANINGFUL_AP = 0.5  # pre-specified: |delta AP| >= 0.5 points and 95% CI excluding 0
METRICS = {"AP": ("all", None), "AP50": ("all", 0.5), "AP75": ("all", 0.75),
           "APs": ("small", None), "APm": ("medium", None), "APl": ("large", None)}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _coco(path_or_dict, iou_type: str = "bbox") -> COCO:
    d = json.loads(Path(path_or_dict).read_text()) if not isinstance(path_or_dict, dict) else path_or_dict
    if iou_type == "segm":  # a box-only object is scored as its rectangle
        for a in d["annotations"]:
            if not a.get("segmentation"):
                x, y, w, h = a["bbox"]
                a["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    with _quiet():
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def gt_signature(gt: dict, iou_type: str) -> str:
    """Hash of what the evaluation actually reads, to evaluate identical files once."""
    keys = ("image_id", "category_id", "bbox", "area", "iscrowd") + (("segmentation",) if iou_type == "segm" else ())
    rows = sorted(json.dumps([a.get(k) for k in keys], sort_keys=True) for a in gt["annotations"])
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _table(gt: COCO, preds: list, iou_type: str):
    with _quiet():
        # loadRes writes into the dicts it gets, so give it fresh copies each time
        if preds:
            dt = gt.loadRes([dict(p) for p in preds])
        else:  # a model that predicts nothing scores 0
            dt = COCO()
            dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"],
                          "annotations": []}
            dt.createIndex()
        E = run_cocoeval(gt, dt, iou_type)
        T = APTable(E)
        E.accumulate()
        E.summarize()
    return T, E.stats


def evaluate_combo(task: dict) -> list[dict]:
    """One model and one IoU type against every ground-truth file."""
    model, iou_type = task["model"], task["iou_type"]
    preds = json.loads(Path(task["pred_file"]).read_text())
    gt0 = _coco(task["baseline"], iou_type)
    T0, stats0 = _table(gt0, preds, iou_type)
    base = {m: T0.ap(area=a, iou=i) for m, (a, i) in METRICS.items()}
    check = abs(base["AP"] - 100 * stats0[0])  # our AP must equal pycocotools'
    W = bootstrap_weights(T0.n_img, task["n_boot"], seed=0) if task["n_boot"] else None
    base_boot = np.array([T0.ap(w) for w in W]) if W is not None else None

    rows, cache = [], {}
    for g in task["gts"]:
        gt_dict = json.loads(Path(g["file"]).read_text())
        sig = gt_signature(gt_dict, iou_type)
        if sig not in cache:
            T1, _ = _table(_coco(gt_dict, iou_type), preds, iou_type)
            vals = {m: T1.ap(area=a, iou=i) for m, (a, i) in METRICS.items()}
            # An interval is only needed where the rule could call the effect
            # meaningful (|dAP| >= 0.5); smaller effects fail the rule anyway.
            need = W is not None and abs(vals["AP"] - base["AP"]) >= MEANINGFUL_AP
            boot = np.array([T1.ap(w) for w in W]) - base_boot if need else None
            cache[sig] = (vals, boot)
        vals, boot = cache[sig]
        row = {"model": model, "iou_type": iou_type, "variant": g["variant"], "path": g["path"],
               "component": g["component"], "objects_changed": g["changed"],
               "check_vs_pycocotools": round(check, 6)}
        for m in METRICS:
            row[f"{m}_original"] = round(base[m], 3)
            row[f"{m}_converted"] = round(vals[m], 3)
            row[f"d{m}"] = round(vals[m] - base[m], 3)
        row["dAP_ci_low"] = row["dAP_ci_high"] = float("nan")
        row["meaningful"] = False
        if boot is not None:
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            row["dAP_ci_low"], row["dAP_ci_high"] = round(float(lo), 3), round(float(hi), 3)
            row["meaningful"] = bool(abs(row["dAP"]) >= MEANINGFUL_AP and (lo > 0 or hi < 0))
        rows.append(row)
    return rows
