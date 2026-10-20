"""Match converted objects to originals and measure what changed."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import (ann_to_mask, box_ioa, box_iou_matrix, is_axis_rectangle,
                       mask_iou, polygons_to_mask)

MATCH_IOU = 0.5      # first pass: a converted object overlapping its original this much
WEAK_IOU = 0.1       # second pass, same class only: a badly moved object is still that object
FRAGMENT_IOA = 0.9   # an extra object this much inside an original is a fragment
FRAGMENT_MARGIN = 2.0  # px tolerance around the original box for that test
CHANGED_IOU = 0.9    # below this, geometry counts as materially changed


def _match(orig, conv, conv_free, min_iou, same_class=False):
    """Hungarian matching on box IoU among still-free converted objects."""
    idx = [j for j in range(len(conv)) if conv_free[j]]
    if not len(orig) or not idx:
        return {}
    iou = box_iou_matrix(np.asarray([o["box"] for o in orig], float),
                         np.asarray([conv[j]["box"] for j in idx], float))
    if same_class:
        same = np.array([[conv[j].get("category_id") == o["category_id"] for j in idx] for o in orig])
        iou = np.where(same, iou, 0.0)
    r, c = linear_sum_assignment(-iou)
    return {i: (idx[k], float(iou[i, k])) for i, k in zip(r, c) if iou[i, k] >= min_iou}


def _clip(box, w, h):
    return [min(max(box[0], 0.0), w), min(max(box[1], 0.0), h),
            min(max(box[2], 0.0), w), min(max(box[3], 0.0), h)]


def _truthy(v) -> bool:
    """Tool-written flags arrive as text ("True", "1", "0", "False") or None."""
    return str(v).strip().lower() in ("1", "true", "yes")


def measure_image(originals: list[dict], converted: list[dict], kind: str,
                  h: int, w: int) -> tuple[list[dict], dict]:
    """Return one row per original object and per-image extra counts.

    Converted boxes are clipped to the image first, as training loaders do,
    so a polygon vertex outside the image does not count as a change.
    """
    converted = [dict(c, box=_clip(c["box"], w, h)) for c in converted]
    free = [True] * len(converted)
    rows: list[dict] = []

    noncrowd = [o for o in originals if not o["iscrowd"]]
    crowd = [o for o in originals if o["iscrowd"]]

    m1 = _match(noncrowd, converted, free, MATCH_IOU)
    for _, (j, _) in m1.items():
        free[j] = False
    rest = [i for i in range(len(noncrowd)) if i not in m1]
    weak = _match([noncrowd[i] for i in rest], converted, free, WEAK_IOU, same_class=True)
    for k, (j, iou) in weak.items():
        m1[rest[k]] = (j, iou)
        free[j] = False
    m2 = _match(crowd, converted, free, MATCH_IOU)
    for _, (j, _) in m2.items():
        free[j] = False

    # Leftover converted objects that sit inside an original of the same class
    # are fragments of it (one object written as several); others are additions.
    # Crowd regions count too: a tool can write one crowd region as many pieces.
    fragments: dict[int, list[int]] = {}        # non-crowd original -> pieces
    crowd_frag: dict[int, list[int]] = {}       # crowd original -> pieces
    added: list[dict] = []
    candidates = [(False, i, o) for i, o in enumerate(noncrowd)] + \
                 [(True, i, o) for i, o in enumerate(crowd)]
    orig_masks: dict[tuple[bool, int], np.ndarray] = {}

    def _orig_mask(key, o):
        if key not in orig_masks:
            orig_masks[key] = (ann_to_mask(o["ann"], h, w) if o["seg_type"] != "none"
                               else polygons_to_mask([_box_poly(o["box"])], h, w))
        return orig_masks[key]

    for j in range(len(converted)):
        if not free[j]:
            continue
        qualified = []
        for is_crowd, i, o in candidates:
            if converted[j].get("category_id") != o["category_id"]:
                continue
            grown = np.asarray(o["box"], float) + [-FRAGMENT_MARGIN, -FRAGMENT_MARGIN,
                                                   FRAGMENT_MARGIN, FRAGMENT_MARGIN]
            ioa = box_ioa(np.asarray(converted[j]["box"]), grown)
            if ioa >= FRAGMENT_IOA:
                qualified.append((ioa, is_crowd, i, o))
        best_key = None
        if len(qualified) == 1:
            best_key = (qualified[0][1], qualified[0][2])
        elif qualified:
            # Several originals contain the piece's box (a crowd region around a person, or
            # overlapping objects of one class): give it to the one whose mask covers most of
            # the piece; ties go to the larger box overlap, then to a non-crowd original.
            c = converted[j]
            piece = polygons_to_mask([c["poly"] if c.get("poly") is not None else _box_poly(c["box"])], h, w)
            area = max(int(piece.sum()), 1)
            scored = [(float(np.logical_and(piece, _orig_mask((ic, i), o)).sum()) / area, ioa, not ic, ic, i)
                      for ioa, ic, i, o in qualified]
            scored.sort(reverse=True)
            best_key = (scored[0][3], scored[0][4])
        if best_key is not None:
            (crowd_frag if best_key[0] else fragments).setdefault(best_key[1], []).append(j)
            free[j] = False
        else:
            c = converted[j]
            added.append({"category_id": c.get("category_id"), "kind": c["kind"], "index": j,
                          "x1": c["box"][0], "y1": c["box"][1], "x2": c["box"][2], "y2": c["box"][3]})
    n_added = len(added)

    # An original with no IoU match but with pieces inside it was split, not
    # dropped: its largest piece becomes the match and the rest stay fragments.
    for i, frag in fragments.items():
        if i in m1:
            continue
        frag.sort(key=lambda k: _area(converted[k]["box"]), reverse=True)
        j = frag.pop(0)
        biou = float(box_iou_matrix(np.asarray([noncrowd[i]["box"]], float),
                                    np.asarray([converted[j]["box"]], float))[0, 0])
        m1[i] = (j, biou)

    for i, o in enumerate(noncrowd):
        row = _base_row(o)
        if i not in m1:
            row.update(status="dropped", material=True)
            rows.append(row)
            continue
        j, biou = m1[i]
        c = converted[j]
        frag = fragments.get(i, [])
        row.update(
            status="matched",
            class_ok=c.get("category_id") == o["category_id"],
            box_iou=biou,
            box_max_dev_px=float(np.max(np.abs(np.asarray(c["box"]) - np.asarray(o["box"])))),
            conv_geom=c["kind"],
            n_fragments=len(frag),
        )
        geom_iou = biou
        if kind == "seg" and o["seg_type"] != "none":
            orig_mask = ann_to_mask(o["ann"], h, w)
            poly = c["poly"] if c["poly"] is not None else _box_poly(c["box"])
            row["mask_iou"] = mask_iou(orig_mask, polygons_to_mask([poly], h, w))
            polys = [poly] + [converted[k]["poly"] or _box_poly(converted[k]["box"]) for k in frag]
            row["mask_iou_union"] = (mask_iou(orig_mask, polygons_to_mask(polys, h, w))
                                     if frag else row["mask_iou"])
            row["rect_fallback"] = bool(
                is_axis_rectangle(poly, o["box"]) and not _orig_is_rectangle(o))
            geom_iou = row["mask_iou"]
        row["geom_changed"] = bool(geom_iou < CHANGED_IOU)
        row["material"] = bool(not row["class_ok"] or row["geom_changed"] or frag)
        rows.append(row)

    for i, o in enumerate(crowd):
        row = _base_row(o)
        pieces = ([m2[i][0]] if i in m2 else []) + crowd_frag.get(i, [])
        if pieces:
            # kept as ordinary objects unless every piece carries a standard flag
            flagged = all(str(converted[k].get("difficult")) == "1" for k in pieces)
            row.update(status="crowd_kept", n_fragments=len(pieces) - 1,
                       crowd_attr_kept=any(_truthy(converted[k].get("attrs", {}).get("is_crowd")) for k in pieces),
                       material=not flagged)
        else:
            row.update(status="crowd_dropped", material=False)
        rows.append(row)

    # Which converted objects belong to which original (indices into `converted`,
    # whose boxes are clipped to the image). Used to build converted ground truth.
    assign = {
        "noncrowd": {o["ann_id"]: {"primary": m1[i][0], "fragments": list(fragments.get(i, []))}
                     for i, o in enumerate(noncrowd) if i in m1},
        "dropped": [o["ann_id"] for i, o in enumerate(noncrowd) if i not in m1],
        "crowd": {o["ann_id"]: ([m2[i][0]] if i in m2 else []) + crowd_frag.get(i, [])
                  for i, o in enumerate(crowd)},
        "added": [a["index"] for a in added],
        "converted": converted,
    }
    for a in added:
        a.pop("index")
    return rows, {"n_added": n_added, "added": added, "assign": assign}


def _base_row(o: dict) -> dict:
    return {k: o[k] for k in ("ann_id", "image_id", "category_id", "iscrowd", "seg_type",
                              "n_parts", "has_holes", "size", "touches_border")}


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_poly(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _orig_is_rectangle(o: dict) -> bool:
    seg = o["ann"].get("segmentation")
    return isinstance(seg, list) and len(seg) == 1 and is_axis_rectangle(seg[0], o["box"])
