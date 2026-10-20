"""Stage 2 analysis: training-label effects, paired by seed.

Pre-specified rule (2026-10-15, before any training run): a training
condition differs meaningfully from its reference if the mean seed-paired
difference in AP on the original val2017 ground truth is at least 0.5 AP points
in magnitude and its 95% t-interval (paired by seed) excludes 0. Seeds are
paired because the same seed gives the same initialization and data order;
only the labels differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import METRICS, _coco, _table
from .stage2 import CONDITIONS, REFERENCE_OF

MEANINGFUL_AP = 0.5


def _runs(project: Path) -> list[dict]:
    runs = []
    for d in sorted(project.glob("*_s*")):
        f = d / "done.json"
        if f.exists():
            runs.append(json.loads(f.read_text()))
    return runs


def evaluate_runs(project: Path, gt_files: dict[str, Path]) -> pd.DataFrame:
    """AP of every finished run against each ground-truth file (name -> path)."""
    rows = []
    gts = {}
    for run in _runs(project):
        for iou_type, pfile in run["predictions"].items():
            preds = json.loads(Path(pfile).read_text())
            for gname, gpath in gt_files.items():
                key = (gname, iou_type)
                if key not in gts:
                    gts[key] = _coco(gpath, iou_type)
                T, _ = _table(gts[key], preds, iou_type)
                row = {"condition": run["condition"], "task": run["task"], "seed": run["seed"],
                       "iou_type": iou_type, "eval_gt": gname}
                row.update({m: T.ap(area=a, iou=i) for m, (a, i) in METRICS.items()})
                rows.append(row)
    return pd.DataFrame(rows)


# Not a conversion effect: the same faithful Ultralytics labels written as boxes vs polygons.
# It measures how much the label *format* alone changes a YOLO detector, which is the
# confound that motivated pairing supervision with det_reference_poly.
CONTROLS = {"det_reference_poly": "det_reference"}


def _paired(gc: pd.DataFrame, ref: pd.DataFrame, row: dict) -> None:
    seeds = sorted(set(gc.index) & set(ref.index))
    d = (gc.loc[seeds, "AP"] - ref.loc[seeds, "AP"]).to_numpy()
    row["paired_seeds"] = len(seeds)
    row["dAP_mean"] = round(float(d.mean()), 3) if len(d) else np.nan
    for m in ("AP50", "AP75", "APs", "APm", "APl"):
        row[f"d{m}_mean"] = round(float((gc.loc[seeds, m] - ref.loc[seeds, m]).mean()), 3) if len(d) else np.nan
    row["dAP_ci_low"] = row["dAP_ci_high"] = np.nan
    if len(d) > 1:
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - half, d.mean() + half
        row["dAP_ci_low"], row["dAP_ci_high"] = round(float(lo), 3), round(float(hi), 3)
        row["meaningful"] = bool(abs(d.mean()) >= MEANINGFUL_AP and (lo > 0 or hi < 0))
    else:
        row["meaningful"] = None  # not decidable with one seed


def compare(res: pd.DataFrame, eval_gt: str = "original") -> pd.DataFrame:
    """Each condition vs its own reference (REFERENCE_OF), paired by seed, on one evaluation truth.

    Reference conditions get a row with their mean and SD only. det_reference_poly also gets a
    `control` row against det_reference (label format alone, no conversion defect).
    """
    r = res[res.eval_gt == eval_gt]
    out = []
    for iou_type, g in r.groupby("iou_type"):
        by = {c: gc.set_index("seed") for c, gc in g.groupby("condition")}
        for cond, gc in by.items():
            row = {"task": CONDITIONS[cond][0] if cond in CONDITIONS else "", "iou_type": iou_type,
                   "condition": cond, "reference": REFERENCE_OF.get(cond) or "", "role": "",
                   "eval_gt": eval_gt, "seeds": len(gc)}
            for m in METRICS:
                row[f"{m}_mean"] = round(gc[m].mean(), 3)
                row[f"{m}_sd"] = round(gc[m].std(ddof=1), 3) if len(gc) > 1 else np.nan
            ref = REFERENCE_OF.get(cond)
            if ref is None:
                row["role"] = "reference"
                out.append(row)
                if cond in CONTROLS and CONTROLS[cond] in by:
                    ctrl = dict(row, reference=CONTROLS[cond], role="control")
                    _paired(gc, by[CONTROLS[cond]], ctrl)
                    out.append(ctrl)
                continue
            row["role"] = "conversion"
            if ref in by:
                _paired(gc, by[ref], row)
            else:
                row["paired_seeds"] = 0
                row["meaningful"] = None
            out.append(row)
    return pd.DataFrame(out)


def analyze(project: Path, val_gt: Path, out_dir: Path, stage1_gt: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_files = {"original": val_gt}
    if stage1_gt is not None:
        # train/eval mismatch: evaluate against val ground truth converted by each tool
        for p in ("ultralytics_yolo_det", "ultralytics_yolo_seg", "supervision_yolo", "datumaro_yolo_det"):
            f = stage1_gt / f"original__{p}__full.json"
            if f.exists():
                gt_files[f"converted:{p}"] = f
    res = evaluate_runs(project, gt_files)
    res.to_csv(out_dir / "stage2_runs.csv", index=False)
    cmp = compare(res)
    cmp.to_csv(out_dir / "stage2_comparison.csv", index=False)
    parts = ["# Stage 2: effect of converted training labels\n",
             "All models are evaluated on the original COCO val2017 ground truth unless stated. "
             "dAP = condition - its reference, paired by seed. Rule: meaningful if |mean dAP| >= "
             f"{MEANINGFUL_AP} and the 95% paired t-interval excludes 0.\n\n"
             "Pairing (protocol amendment of 2026-10-15): supervision writes polygon labels, so its "
             "detection run is compared with det_reference_poly (Ultralytics' faithful polygon labels); "
             "Datumaro writes boxes, so it is compared with det_reference. The `control` row "
             "(det_reference_poly vs det_reference) is the effect of label format alone, not a "
             "conversion defect.\n",
             "\n## Comparison with the reference\n\n" + cmp.to_markdown(index=False) + "\n"]
    if len(gt_files) > 1:
        piv = res.pivot_table(index=["condition", "iou_type"], columns="eval_gt", values="AP", aggfunc="mean").round(2)
        parts.append("\n## Train/evaluation matrix (mean AP over seeds)\n\nRows: training labels. "
                     "Columns: evaluation ground truth.\n\n" + piv.to_markdown() + "\n")
    (out_dir / "report_stage2.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Report: {out_dir / 'report_stage2.md'}")
