"""A small COCO file whose objects cover the cases converters handle differently."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

CATEGORIES = [  # real COCO ids and names, so the COCO-80 convention applies
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 17, "name": "cat", "supercategory": "animal"},
    {"id": 90, "name": "toothbrush", "supercategory": "indoor"},
]


def _rle(mask: np.ndarray) -> dict:
    flat = mask.astype(np.uint8).flatten(order="F")
    change = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate(([0], change, [flat.size]))).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": runs}


def _poly_ann(aid, img, cat, polys, h, w):
    rle = mask_utils.merge(mask_utils.frPyObjects(polys, h, w))
    x, y, bw, bh = mask_utils.toBbox(rle).tolist()
    return {"id": aid, "image_id": img, "category_id": cat, "segmentation": polys,
            "area": float(mask_utils.area(rle)), "bbox": [x, y, bw, bh], "iscrowd": 0}


def _mask_ann(aid, img, cat, m, crowd):
    rle = _rle(m)
    enc = mask_utils.frPyObjects(rle, *m.shape)
    x, y, bw, bh = mask_utils.toBbox(enc).tolist()
    return {"id": aid, "image_id": img, "category_id": cat, "segmentation": rle,
            "area": float(mask_utils.area(enc)), "bbox": [x, y, bw, bh], "iscrowd": int(crowd)}


def make_fixture() -> dict:
    images = [
        {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "img2.jpg", "width": 500, "height": 375},
        {"id": 3, "file_name": "img3.jpg", "width": 333, "height": 500},  # no objects
    ]
    anns = []
    # 1 simple single-part polygon (an octagon)
    anns.append(_poly_ann(1, 1, 1, [[130, 100, 170, 100, 200, 140, 200, 210, 170, 250,
                                    130, 250, 100, 210, 100, 140]], 480, 640))
    # 2 one object stored as two separate polygon parts
    anns.append(_poly_ann(2, 1, 3, [[300, 100, 360, 100, 360, 160, 300, 160],
                                    [380, 100, 440, 100, 440, 160, 380, 160]], 480, 640))
    # 3 non-crowd RLE mask with a hole (a ring)
    ring = np.zeros((480, 640), np.uint8)
    ring[300:380, 450:530] = 1
    ring[325:355, 475:505] = 0
    anns.append(_mask_ann(3, 1, 17, ring, crowd=False))
    # 4 crowd region (RLE, iscrowd=1)
    crowd = np.zeros((480, 640), np.uint8)
    crowd[300:400, 10:110] = 1
    anns.append(_mask_ann(4, 1, 1, crowd, crowd=True))
    # 5 a tiny polygon (6 x 6 px)
    anns.append(_poly_ann(5, 2, 90, [[50, 50, 56, 50, 56, 56, 50, 56]], 375, 500))
    # 6 a polygon touching the right and bottom borders
    anns.append(_poly_ann(6, 2, 3, [[440, 300, 500, 300, 500, 375, 440, 375]], 375, 500))
    # 7 a box-only object (empty segmentation)
    anns.append({"id": 7, "image_id": 2, "category_id": 1, "segmentation": [],
                 "area": 4000.0, "bbox": [100.0, 100.0, 50.0, 80.0], "iscrowd": 0})
    return {"info": {"description": "convaudit fixture"}, "licenses": [],
            "images": images, "annotations": anns, "categories": CATEGORIES}
