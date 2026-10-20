"""Generate a COCO-sized synthetic instances file to check runtime and memory.

Not for results: geometry is random. Usage:
    python tests/make_synthetic_coco.py out.json --images 5000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from pycocotools import mask as mask_utils

from fixture import CATEGORIES  # noqa: F401  (kept for reference)
from convaudit.coco_io import COCO80_IDS


def blob(rng, cx, cy, r, n=12):
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = r * rng.uniform(0.6, 1.0, n)
    return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).ravel().round(2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--images", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    images, anns, aid = [], [], 1
    for i in range(1, a.images + 1):
        w, h = int(rng.choice([640, 500, 480])), int(rng.choice([480, 427, 375, 640]))
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": w, "height": h})
        for _ in range(rng.poisson(7)):
            r = float(rng.choice([4, 12, 40, 120]) * rng.uniform(0.7, 1.3))
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            polys = [blob(rng, cx, cy, r)]
            if rng.random() < 0.15:  # some objects in two parts
                polys.append(blob(rng, cx + 2.5 * r, cy, r / 2))
            crowd = rng.random() < 0.01
            rles = mask_utils.frPyObjects(polys, h, w)
            rle = mask_utils.merge(rles)
            area = float(mask_utils.area(rle))
            if area < 1:
                continue
            seg = polys
            if crowd:
                m = mask_utils.decode(rle)
                flat = m.flatten(order="F")
                ch = np.flatnonzero(np.diff(flat)) + 1
                runs = np.diff(np.concatenate(([0], ch, [flat.size]))).tolist()
                seg = {"size": [h, w], "counts": ([0] + runs) if flat[0] else runs}
            anns.append({"id": aid, "image_id": i, "category_id": int(rng.choice(COCO80_IDS)),
                         "segmentation": seg, "area": area,
                         "bbox": mask_utils.toBbox(rle).tolist(), "iscrowd": int(crowd)})
            aid += 1
    cats = [{"id": c, "name": f"class_{c}", "supercategory": "x"} for c in COCO80_IDS]
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(a.out, "w"))
    print(len(images), "images,", len(anns), "annotations")


if __name__ == "__main__":
    main()
