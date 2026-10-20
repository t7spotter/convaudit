"""Reading, sharding and preparing COCO instance files."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import ann_to_mask, mask_components, mask_has_holes

# The 80 category ids used by COCO 2017 detection, in their official order.
# Index i in this list is the class index of the common "COCO-80" convention.
COCO80_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
]
assert len(COCO80_IDS) == 80

SMALL, MEDIUM = 32 ** 2, 96 ** 2  # COCO area thresholds


def load_coco(path: str | Path) -> dict:
    """Load a COCO instances file. Optional fields some releases omit (iscrowd, area,
    bbox) are filled in the standard way, and the count is printed, so every tool
    receives a complete COCO file."""
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)
    sizes = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
    filled = 0
    for a in coco["annotations"]:
        if "iscrowd" not in a:
            a["iscrowd"] = 0
            filled += 1
        if "area" not in a or "bbox" not in a:
            from pycocotools import mask as mu
            h, w = sizes[a["image_id"]]
            seg = a["segmentation"]
            if isinstance(seg, list):
                rle = mu.merge(mu.frPyObjects(seg, h, w))
            elif isinstance(seg.get("counts"), list):
                rle = mu.frPyObjects(seg, h, w)
            else:
                rle = seg
            a.setdefault("area", float(mu.area(rle)))
            a.setdefault("bbox", [float(v) for v in mu.toBbox(rle)])
            filled += 1
    if filled:
        print(f"   note: filled missing iscrowd/area/bbox on {filled} annotations", flush=True)
    return coco


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def limit_images(coco: dict, n: int | None) -> dict:
    """Keep the first n images by id (deterministic) and their annotations."""
    if not n:
        return coco
    keep = sorted(coco["images"], key=lambda im: im["id"])[:n]
    ids = {im["id"] for im in keep}
    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = keep
    out["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
    return out


def shards(coco: dict, size: int):
    """Yield sub-datasets of at most `size` images, each with all categories."""
    images = sorted(coco["images"], key=lambda im: im["id"])
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    base = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    for i in range(0, len(images), size):
        chunk = images[i:i + size]
        s = dict(base)
        s["images"] = chunk
        s["annotations"] = [a for im in chunk for a in by_img.get(im["id"], [])]
        yield s


def _uncompressed_rle(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE: column-major run lengths, starting with zeros."""
    flat = mask.astype(np.uint8).flatten(order="F")
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:  # counts must start with a run of zeros
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": runs}


def make_rle_variant(coco: dict) -> dict:
    """Copy of `coco` where every non-crowd polygon is stored as RLE instead.

    This mimics datasets exported by tools that save masks as RLE (brush and
    SAM masks, for example). The pixels are unchanged: each RLE is the exact
    rasterization of the original polygon, so any difference a converter
    produces on this variant comes from how it handles RLE.
    """
    out = copy.deepcopy(coco)
    size = {im["id"]: (im["height"], im["width"]) for im in out["images"]}
    for a in out["annotations"]:
        seg = a.get("segmentation")
        if a.get("iscrowd") or not isinstance(seg, list) or not seg:
            continue
        h, w = size[a["image_id"]]
        a["segmentation"] = _uncompressed_rle(ann_to_mask(a, h, w))
    return out


def write_coco(coco: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f)


def write_stub_images(coco: dict, images_dir: str | Path) -> None:
    """Write black JPEGs of the right size for tools that open every image.

    Annotation geometry never depends on pixel values, so stubs are enough to
    let such tools read image sizes. Use real images for the final study.
    """
    images_dir = Path(images_dir)
    for im in coco["images"]:
        p = images_dir / im["file_name"]
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.zeros((im["height"], im["width"], 3), np.uint8))


def object_table(coco: dict) -> dict[int, list[dict]]:
    """Originals per image with the descriptive flags used for stratification."""
    imgs = {im["id"]: im for im in coco["images"]}
    out: dict[int, list[dict]] = {im_id: [] for im_id in imgs}
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        h, w = im["height"], im["width"]
        x, y, bw, bh = a["bbox"]
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            seg_type = "rle"
        elif isinstance(seg, list) and seg:
            seg_type = "polygon"
        else:
            seg_type = "none"
        n_parts, holes = 0, False
        if seg_type == "polygon":
            n_parts = len(seg)
        if seg_type != "none":
            m = ann_to_mask(a, h, w)
            if seg_type == "rle":
                n_parts = mask_components(m)
            holes = mask_has_holes(m)
        area = float(a.get("area", bw * bh))
        out[a["image_id"]].append({
            "ann_id": a["id"],
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "iscrowd": int(bool(a.get("iscrowd", 0))),
            "box": [x, y, x + bw, y + bh],
            "seg_type": seg_type,
            "n_parts": n_parts,
            "has_holes": bool(holes),
            "size": "small" if area < SMALL else "medium" if area < MEDIUM else "large",
            "touches_border": bool(x <= 0.5 or y <= 0.5 or x + bw >= w - 0.5 or y + bh >= h - 0.5),
            "ann": a,
        })
    return out
