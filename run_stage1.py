"""Stage 1: how much does converted ground truth change measured AP?

Predictions of fixed, pretrained models are computed once. Each conversion
path's output is rebuilt as a COCO file, whole ("full") and one defect type at
a time, and every model is evaluated against the original and every converted
file. Differences come only from the labels.

Example:
    python run_stage1.py --coco annotations/instances_val2017.json --images val2017 --out stage1
    python run_stage1.py --coco annotations/instances_val2017.json --images val2017 --out stage1_quick --limit-images 200 --bootstrap 100
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops run on CPU

import argparse
import shutil
import json
import multiprocessing as mp
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from convaudit import __version__
from convaudit.adapters import PATHS, run_path, tool_version
from convaudit.coco_io import (limit_images, load_coco, make_rle_variant,
                               object_table, sha256_file, shards, write_coco)
from convaudit.evaluate import MEANINGFUL_AP, evaluate_combo
from convaudit.gt_build import COMPONENTS, compose, finalize
from convaudit.measure import measure_image
from convaudit.predict import MODELS, OUTPUTS, predict

SEG_PATHS = {p for p, (_, kind, _) in PATHS.items() if kind == "seg"}


def build_ground_truth(coco: dict, variants: list[str], paths: list[str], gt_dir: Path,
                       images_dir: Path | None, shard_size: int, work: Path) -> list[dict]:
    gt_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"original": coco}
    if "rle" in variants:
        datasets["rle"] = make_rle_variant(coco)
    index = []
    images = {im["id"]: im for im in coco["images"]}
    for variant in variants:
        data = datasets[variant]
        table = object_table(data)
        for path in paths:
            print(f"[ground truth] {variant} / {path} ...", flush=True)
            with_seg = path in SEG_PATHS
            anns = {c: [] for c in COMPONENTS}
            changed = {c: 0 for c in COMPONENTS}
            for k, shard in enumerate(shards(data, shard_size)):
                result, meta = run_path(path, shard, work / variant / path / f"shard{k:04d}", images_dir)
                for im in shard["images"]:
                    objs, _ = result[im["file_name"]]
                    _, extra = measure_image(table[im["id"]], objs, meta["kind"], im["height"], im["width"])
                    for c in COMPONENTS:
                        a, n = compose(c, images[im["id"]], table[im["id"]], extra["assign"], with_seg)
                        anns[c] += a
                        changed[c] += n
            for c in COMPONENTS:
                if c != "full" and changed[c] == 0:
                    continue  # nothing of this kind changed: identical to original
                f = gt_dir / f"{variant}__{path}__{c}.json"
                write_coco(finalize(coco, anns[c]), f)
                index.append({"variant": variant, "path": path, "component": c, "changed": changed[c],
                              "annotations": len(anns[c]), "with_seg": with_seg, "file": str(f)})
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True, help="folder with the COCO images (val2017)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--paths", default="all")
    ap.add_argument("--variants", default="original,rle")
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples (0 = none)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="parallel evaluations; each needs ~1-2 GB of memory")
    ap.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--predictions-from", default=None,
                    help="predictions folder of an earlier run on the same images (e.g. stage1/predictions); "
                         "reused instead of running the models again")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images)
    models = [m.strip() for m in args.models.split(",")]
    paths = list(PATHS) if args.paths == "all" else [p.strip() for p in args.paths.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    coco = limit_images(load_coco(args.coco), args.limit_images)
    missing = [im["file_name"] for im in coco["images"] if not (images_dir / im["file_name"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images not found in {images_dir} (first: {missing[0]})")
    t_start = time.time()

    # 1. predictions (computed once, resumable)
    pred_files = {}
    if args.predictions_from:
        # Predictions depend only on the images, not on the ground truth, so a run on
        # another annotation file of the same images (COCO-ReM) or on other tools reuses them.
        src = Path(args.predictions_from)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        for m in models:
            for f in src.glob(f"{m}.*"):
                dst = out / "predictions" / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
    for m in models:
        print(f"[predictions] {m} ...", flush=True)
        pred_files[m] = predict(m, coco, images_dir, out / "predictions", args.device)

    # 2. ground truth: original and converted
    baseline = out / "gt" / "original.json"
    write_coco(coco, baseline)
    index = build_ground_truth(coco, variants, paths, out / "gt", images_dir, args.shard_size, out / "work")
    pd.DataFrame(index).to_csv(out / "gt" / "index.csv", index=False)

    # 3. evaluation, one task per model and IoU type, in parallel
    tasks = []
    for m in models:
        for t in OUTPUTS[m]:
            gts = [g for g in index if t == "bbox" or g["with_seg"]]
            if not gts:  # e.g. only box-format paths: nothing to score masks against
                continue
            tasks.append({"model": m, "iou_type": t, "pred_file": str(pred_files[m][t]),
                          "baseline": str(baseline), "gts": gts, "n_boot": args.bootstrap})
    print(f"[evaluation] {len(tasks)} model/IoU combinations on {args.workers} workers ...", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        for r in pool.imap_unordered(evaluate_combo, tasks):
            rows += r
            print(f"   done: {r[0]['model']} / {r[0]['iou_type']}", flush=True)

    res = pd.DataFrame(rows).sort_values(["iou_type", "model", "variant", "path", "component"])
    res.to_csv(out / "stage1_results.csv", index=False)
    write_report(res, out, args)
    manifest = {
        "convaudit": __version__, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes": round((time.time() - t_start) / 60, 1), "python": sys.version.split()[0],
        "platform": platform.platform(), "input_sha256": sha256_file(args.coco),
        "images": len(coco["images"]), "args": vars(args),
        "tools": {p: tool_version(p) for p in ("ultralytics", "supervision", "datumaro", "fiftyone", "pycocotools",
                                                "torch", "torchvision")},
        "max_check_vs_pycocotools": float(res["check_vs_pycocotools"].max()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone in {manifest['minutes']} min. Report: {out / 'report.md'}")
    return 0


def write_report(res: pd.DataFrame, out: Path, args) -> None:
    cols = ["model", "variant", "path", "component", "objects_changed", "AP_original", "AP_converted",
            "dAP", "dAP_ci_low", "dAP_ci_high", "meaningful", "dAP50", "dAP75", "dAPs", "dAPm", "dAPl"]
    cols = [c for c in cols if c in res]
    parts = ["# Stage 1: effect of converted ground truth on measured AP\n",
             f"Pre-specified rule: a conversion effect is meaningful if |dAP| >= {MEANINGFUL_AP} AP points "
             "and the 95% paired bootstrap interval (images resampled) excludes 0. dAP = converted - original; "
             "predictions are identical in both.\n",
             f"Largest difference between this harness's AP and pycocotools' official AP: "
             f"{res['check_vs_pycocotools'].max():.6f} points (should be 0).\n"]
    for iou_type in ("bbox", "segm"):
        r = res[res.iou_type == iou_type]
        if r.empty:
            continue
        full = r[r.component == "full"]
        piv = full.pivot_table(index=["variant", "path"], columns="model", values="dAP").round(2)
        parts.append(f"\n## {iou_type}: dAP of the full conversion, per model\n\n" + piv.to_markdown() + "\n")
        parts.append(f"\n## {iou_type}: all components\n\n" + r[cols].to_markdown(index=False) + "\n")
    (out / "report.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
