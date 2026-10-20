"""Independent readers for converter output.

These parsers implement the published format conventions directly and import
nothing from the tools under test, so a tool cannot hide its own changes by
reading back what it wrote.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def parse_yolo_txt(path: Path, w: int, h: int) -> tuple[list[dict], int]:
    """Read one YOLO label file.

    Returns (objects, n_malformed). A line with 4 coordinates is a box
    (cx cy bw bh, normalized). A line with an even number >= 6 of coordinates
    is a polygon (x1 y1 x2 y2 ..., normalized). Anything else is malformed.
    """
    objs, bad = [], 0
    if not path.exists():
        return objs, bad
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if not tok:
            continue
        try:
            cls = int(float(tok[0]))
            vals = [float(t) for t in tok[1:]]
        except ValueError:
            bad += 1
            continue
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            objs.append({"cls": cls, "kind": "box", "poly": None,
                         "box": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h]})
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            poly = [v * (w if i % 2 == 0 else h) for i, v in enumerate(vals)]
            xs, ys = poly[0::2], poly[1::2]
            objs.append({"cls": cls, "kind": "poly", "poly": poly,
                         "box": [min(xs), min(ys), max(xs), max(ys)]})
        else:
            bad += 1
    return objs, bad


def read_names_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_yaml_names(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def parse_voc_xml(path: Path, convention: str = "devkit") -> tuple[list[dict], int]:
    """Read one Pascal VOC XML file into 0-based pixel-edge boxes.

    convention="devkit": VOC boxes are 1-based inclusive pixel indices, so
    x1 = xmin - 1, y1 = ymin - 1, x2 = xmax, y2 = ymax. This is how
    Detectron2's VOC loader reads them.
    convention="raw": take the four numbers as written.
    """
    if convention not in ("devkit", "raw"):
        raise ValueError(convention)
    objs, bad = [], 0
    if not path.exists():
        return objs, bad
    root = ET.parse(path).getroot()
    for o in root.findall("object"):
        try:
            bb = o.find("bndbox")
            box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            if convention == "devkit":
                box[0] -= 1.0
                box[1] -= 1.0
            attrs = {a.findtext("name"): a.findtext("value")
                     for a in o.findall("attributes/attribute")}
            if o.findtext("iscrowd") is not None:  # FiftyOne writes COCO's flag as its own tag
                attrs.setdefault("is_crowd", o.findtext("iscrowd"))
            objs.append({"name": o.findtext("name"), "kind": "box", "poly": None,
                         "box": box, "difficult": o.findtext("difficult"),
                         "attrs": attrs})
        except (AttributeError, TypeError, ValueError):
            bad += 1
    return objs, bad
