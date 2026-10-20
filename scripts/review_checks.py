"""Extra counts asked for by the manuscript review (run once on the Mac, from the convaudit folder).

    python scripts/review_checks.py --coco annotations/instances_val2017.json \
        --rem ~/Downloads/instances_valrem.json --objects results/objects.csv \
        --rem-objects results_rem/objects.csv

Prints plain-text tables; nothing is written. It answers four questions:
1. Which COCO objects does supervision's YOLO path split? Multi-part objects, and how many
   single-polygon objects, and why (does supervision's own rasterization break them apart?).
2. How common are masks with holes (COCO polygons, the RLE-stored copy, COCO-ReM), and how much
   of supervision's geometry change on RLE input do they explain?
3. What share of COCO-ReM boxes are whole-pixel (all four bbox values integers)?
4. Breakdown of material change into its parts for a few paths, to show the categories add up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def sv_raster_parts(seg, h, w):
    """Connected parts of the mask supervision builds from COCO polygons (round + cv2.fillPoly)."""
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        v = np.round(np.asarray(s, float).reshape(-1, 2)).astype(np.int32)
        if len(v) >= 3:
            cv2.fillPoly(m, [v], 1)
    n, _ = cv2.connectedComponents(m, connectivity=8)
    return n - 1


def pct(a, b):
    return f"{a} of {b} ({100 * a / b:.2f}%)" if b else f"{a} of 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--rem", required=True)
    ap.add_argument("--objects", required=True, help="objects.csv of the COCO audit (variants original and rle)")
    ap.add_argument("--rem-objects", required=True, help="objects.csv of the COCO-ReM audit")
    a = ap.parse_args()

    coco = json.loads(Path(a.coco).read_text())
    rem = json.loads(Path(a.rem).read_text())
    ann = {x["id"]: x for x in coco["annotations"]}
    img = {x["id"]: x for x in coco["images"]}
    obj = pd.read_csv(a.objects, low_memory=False)
    robj = pd.read_csv(a.rem_objects, low_memory=False)
    print("objects.csv columns:", list(obj.columns))
    print("variants:", sorted(obj.variant.unique()), "paths:", sorted(obj.path.unique()))

    # 1. splits on supervision -> YOLO, original COCO
    print("\n== 1. Split objects, supervision -> YOLO, COCO val2017 original (non-crowd)")
    g = obj[(obj.variant == "original") & (obj.path == "supervision_yolo") & (obj.iscrowd == 0)]
    split = g[(g.status == "matched") & (g.n_fragments.fillna(0) > 0)]
    multi = g[g.n_parts > 1]
    print("non-crowd objects:", len(g), "| multi-part:", pct(len(multi), len(g)))
    print("split:", pct(len(split), len(g)))
    print("  split and multi-part:", len(split[split.n_parts > 1]),
          "| split and single-part:", len(split[split.n_parts <= 1]))
    print("  multi-part objects that are split:", pct(int((multi.ann_id.isin(split.ann_id)).sum()), len(multi)))
    single = split[split.n_parts <= 1]
    parts = []
    for aid in single.ann_id:
        x = ann[int(aid)]
        im = img[x["image_id"]]
        parts.append(sv_raster_parts(x["segmentation"], im["height"], im["width"]))
    parts = pd.Series(parts, dtype=float)
    if len(parts):
        print("  single-part split objects, parts of supervision's own raster mask:")
        print("   ", parts.value_counts().sort_index().to_dict())
        print("  single-part split objects by size:", single["size"].value_counts().to_dict())
        print("  single-part split objects with holes:", int(single.has_holes.astype(str).str.lower().eq("true").sum()))
        print("  fragments per single-part split object:", single.n_fragments.describe().round(2).to_dict())

    # 2. holes and geometry on RLE input
    print("\n== 2. Holes and geometry change, supervision -> YOLO (non-crowd, matched)")
    for name, frame, var in [("COCO original", obj, "original"), ("COCO RLE copy", obj, "rle"),
                             ("COCO-ReM", robj, "original")]:
        g = frame[(frame.variant == var) & (frame.path == "supervision_yolo") & (frame.iscrowd == 0)]
        holes = g.has_holes.astype(str).str.lower().eq("true")
        m = g[g.status == "matched"]
        gc = m.geom_changed.astype(str).str.lower().eq("true")
        mh = m.has_holes.astype(str).str.lower().eq("true")
        print(f"{name}: with holes {pct(int(holes.sum()), len(g))}; geometry changed {pct(int(gc.sum()), len(g))}; "
              f"of which with holes {int((gc & mh).sum())}; median mask IoU {m.mask_iou.median():.3f}")
        print(f"   geometry changed by size: {m[gc]['size'].value_counts().to_dict()}")

    # 3. whole-pixel boxes in COCO-ReM and COCO
    print("\n== 3. Whole-pixel boxes (all four bbox values integers), non-crowd")
    for name, d in [("COCO val2017", coco), ("COCO-ReM", rem)]:
        b = [x["bbox"] for x in d["annotations"] if not x.get("iscrowd")]
        whole = sum(all(float(v).is_integer() for v in bb) for bb in b)
        print(f"{name}: {pct(whole, len(b))}")

    # 4. decomposition of material change
    print("\n== 4. Material change decomposed (all objects; first matching reason counts)")
    for name, frame, var, path in [("COCO", obj, "original", "supervision_yolo"),
                                   ("COCO", obj, "original", "supervision_voc"),
                                   ("COCO RLE", obj, "rle", "ultralytics_yolo_seg"),
                                   ("COCO-ReM", robj, "original", "supervision_yolo")]:
        g = frame[(frame.variant == var) & (frame.path == path)]
        t = lambda s: s.astype(str).str.lower().eq("true")  # noqa: E731
        reason = np.select(
            [g.status.eq("dropped"), g.status.eq("crowd_kept") & t(g.material),
             g.status.eq("crowd_dropped"), g.status.eq("crowd_kept"),
             g.class_ok.astype(str).str.lower().eq("false"), g.n_fragments.fillna(0) > 0, t(g.geom_changed)],
            ["dropped", "crowd written as object", "crowd dropped (not counted)", "crowd kept, flagged",
             "class changed", "split", "geometry only"], default="unchanged")
        vc = pd.Series(reason).value_counts()
        print(f"{name} {path}: {len(g)} objects; material {pct(int(t(g.material).sum()), len(g))}")
        print("   ", vc.to_dict())


if __name__ == "__main__":
    main()
