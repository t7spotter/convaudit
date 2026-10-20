"""Build converted ground truth as COCO files, whole and one defect at a time.

For a conversion path, the audit tells us which converted objects belong to
which original. From that we write:

  full      every object exactly as the converter wrote it
  crowd     only the converter's handling of crowd regions (dropped, or
            written as ordinary objects); everything else original
  split     only objects written as several pieces; everything else original
  geometry  only the changed geometry of one-to-one matches (coordinate
            conventions, mask outlines, masks replaced by boxes), keeping the
            original area; everything else original
  area      only the area field recomputed from what was written (box formats
            cannot store mask area, so a rebuilt COCO file uses box area, which
            moves objects between COCO's small/medium/large groups)
  dropped   only objects the converter lost

Converted objects get iscrowd=0, a bbox from their geometry, a polygon
segmentation where the output has one, and an area computed from what was
written (mask area for polygons, box area for boxes), as a COCO file rebuilt
from that output would.
"""

from __future__ import annotations

import copy

from pycocotools import mask as mask_utils

COMPONENTS = ("full", "crowd", "split", "geometry", "area", "dropped")


def _poly_area(poly, h, w) -> float:
    return float(mask_utils.area(mask_utils.merge(mask_utils.frPyObjects([poly], h, w))))


def converted_ann(c: dict, image_id: int, h: int, w: int, with_seg: bool,
                  category_id: int | None = None) -> dict:
    x1, y1, x2, y2 = c["box"]
    ann = {"image_id": image_id,
           "category_id": category_id if category_id is not None else c.get("category_id"),
           "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "iscrowd": 0}
    if c.get("poly") is not None and len(c["poly"]) >= 6:
        ann["area"] = _poly_area(c["poly"], h, w)
        if with_seg:
            ann["segmentation"] = [list(map(float, c["poly"]))]
    else:
        ann["area"] = float(ann["bbox"][2] * ann["bbox"][3])
        if with_seg:
            ann["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
    return ann


def compose(component: str, image: dict, originals: list[dict], assign: dict,
            with_seg: bool) -> tuple[list[dict], int]:
    """Annotations for one image under one component, and how many objects differ."""
    h, w, im_id = image["height"], image["width"], image["id"]
    conv = assign["converted"]
    out, changed = [], 0

    def orig(o):
        return copy.deepcopy(o["ann"])

    def conv_ann(j, cat=None):
        return converted_ann(conv[j], im_id, h, w, with_seg, cat)

    if component == "full":
        used = set()
        for o in originals:
            if o["iscrowd"]:
                pieces = assign["crowd"].get(o["ann_id"], [])
                used.update(pieces)
                changed += 1
                out += [conv_ann(j) for j in pieces]
            elif o["ann_id"] in assign["noncrowd"]:
                a = assign["noncrowd"][o["ann_id"]]
                js = [a["primary"]] + a["fragments"]
                used.update(js)
                changed += 1
                out += [conv_ann(j) for j in js]
            else:
                changed += 1  # dropped
        for j in assign["added"]:
            out.append(conv_ann(j))
            changed += 1
        return out, changed

    for o in originals:
        if o["iscrowd"]:
            if component == "crowd":
                pieces = assign["crowd"].get(o["ann_id"], [])
                out += [conv_ann(j) for j in pieces]  # empty list = dropped
                changed += 1
            else:
                out.append(orig(o))
            continue
        a = assign["noncrowd"].get(o["ann_id"])
        if a is None:  # dropped by the converter
            if component == "dropped":
                changed += 1
            else:
                out.append(orig(o))
        elif a["fragments"]:
            if component == "split":
                out += [conv_ann(j) for j in [a["primary"]] + a["fragments"]]
                changed += 1
            else:
                out.append(orig(o))
        elif component == "geometry":
            # keep the original class and area so this isolates coordinates/shape
            c = conv_ann(a["primary"], cat=o["category_id"])
            c["area"] = o["ann"].get("area", c["area"])
            changed += int(c["bbox"] != o["ann"]["bbox"] or with_seg)
            out.append(c)
        elif component == "area":
            c = orig(o)
            c["area"] = conv_ann(a["primary"])["area"]
            changed += int(abs(c["area"] - o["ann"].get("area", c["area"])) > 1e-6)
            out.append(c)
        else:
            out.append(orig(o))
    return out, changed


def finalize(coco: dict, anns: list[dict]) -> dict:
    """A complete COCO file: original images and categories, renumbered annotations."""
    out = {k: v for k, v in coco.items() if k != "annotations"}
    for i, a in enumerate(anns, 1):
        a["id"] = i
    out["annotations"] = anns
    return out
