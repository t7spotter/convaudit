"""The harness must report no change for exact copies and catch known defects."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from convaudit.coco_io import make_rle_variant, object_table
from convaudit.geometry import ann_to_mask, mask_iou
from convaudit.measure import measure_image
from convaudit.parsers import parse_voc_xml, parse_yolo_txt

sys.path.insert(0, str(Path(__file__).parent))
from fixture import make_fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _exact_copy(originals, seg):
    """What a perfect converter would write: same class, same geometry."""
    out = []
    for o in originals:
        if o["iscrowd"]:
            continue
        s = o["ann"]["segmentation"]
        if seg and isinstance(s, list) and len(s) == 1:
            out.append({"category_id": o["category_id"], "kind": "poly", "poly": list(s[0]),
                        "box": list(o["box"])})
        else:
            out.append({"category_id": o["category_id"], "kind": "box", "poly": None,
                        "box": list(o["box"])})
    return out


def _rows(image_id, converted, kind):
    coco = make_fixture()
    table = object_table(coco)
    im = next(i for i in coco["images"] if i["id"] == image_id)
    rows, extra = measure_image(table[image_id], converted, kind, im["height"], im["width"])
    return {r["ann_id"]: r for r in rows}, extra, table[image_id]


def test_exact_copy_has_no_material_change():
    for image_id in (1, 2, 3):
        table = object_table(make_fixture())[image_id]
        rows, extra, _ = _rows(image_id, _exact_copy(table, seg=False), "det")
        assert not any(r["material"] for r in rows.values())
        assert extra["n_added"] == 0


def test_exact_single_polygon_has_mask_iou_one():
    table = object_table(make_fixture())[1]
    rows, _, _ = _rows(1, _exact_copy(table, seg=True), "seg")
    assert rows[1]["mask_iou"] == 1.0 and not rows[1]["material"]


def test_injected_defects_are_detected():
    table = object_table(make_fixture())[1]
    conv = _exact_copy(table, seg=False)
    by_id = {o["ann_id"]: i for i, o in enumerate(o for o in table if not o["iscrowd"])}
    conv[by_id[1]]["box"] = [b + 12 for b in conv[by_id[1]]["box"]]   # shift 12 px
    conv[by_id[2]]["category_id"] = 1                                  # wrong class
    del conv[by_id[3]]                                                 # dropped
    conv.append({"category_id": 1, "kind": "box", "poly": None,        # crowd kept
                 "box": [10, 300, 110, 400]})
    conv.append({"category_id": 3, "kind": "box", "poly": None,        # unrelated extra
                 "box": [600, 10, 630, 40]})
    rows, extra, _ = _rows(1, conv, "det")
    assert rows[1]["geom_changed"] and rows[1]["material"]
    assert rows[2]["class_ok"] is False and rows[2]["material"]
    assert rows[3]["status"] == "dropped"
    assert rows[4]["status"] == "crowd_kept" and rows[4]["material"]
    assert extra["n_added"] == 1


def test_small_shift_is_a_change_not_a_drop():
    table = object_table(make_fixture())[2]
    conv = _exact_copy(table, seg=False)
    tiny = next(c for c in conv if c["category_id"] == 90)
    tiny["box"] = [b + 3 for b in tiny["box"]]       # 6 px object moved 3 px: IoU ~ 0.33
    big = next(c for c in conv if c["category_id"] == 3)
    big["box"] = [big["box"][0], big["box"][1], big["box"][2] + 40, big["box"][3] + 40]  # outside image
    rows, extra, _ = _rows(2, conv, "det")
    assert rows[5]["status"] == "matched" and rows[5]["geom_changed"]
    assert rows[6]["box_iou"] == 1.0  # clipped back to the image edge
    assert extra["n_added"] == 0


def test_crowd_region_written_as_pieces_is_kept_not_added():
    table = object_table(make_fixture())[1]
    conv = _exact_copy(table, seg=False)
    # the crowd region (person, [10,300,110,400]) written as four small pieces
    for x, y in [(10, 300), (60, 300), (10, 350), (60, 350)]:
        conv.append({"category_id": 1, "kind": "box", "poly": None, "box": [x, y, x + 40, y + 40]})
    rows, extra, _ = _rows(1, conv, "det")
    assert rows[4]["status"] == "crowd_kept" and rows[4]["n_fragments"] == 3
    assert extra["n_added"] == 0


def test_piece_inside_crowd_mask_goes_to_crowd_not_to_overlapping_object():
    # An L-shaped person whose box overlaps a crowd region of people. A converted piece lies
    # inside both boxes, but only inside the crowd region's mask: it is a crowd piece, so the
    # person must not count as split.
    from pycocotools import mask as mu
    h = w = 120
    lshape = [0, 0, 100, 0, 100, 20, 20, 20, 20, 100, 0, 100]
    crowd_m = np.zeros((h, w), np.uint8)
    crowd_m[50:100, 50:100] = 1
    rle = mu.encode(np.asfortranarray(crowd_m))
    rle["counts"] = rle["counts"].decode()

    def orig(aid, crowd, seg, box):
        return {"ann_id": aid, "image_id": 1, "category_id": 1, "iscrowd": crowd,
                "seg_type": "rle" if crowd else "polygon", "n_parts": 1, "has_holes": False,
                "size": "large", "touches_border": False, "box": box,
                "ann": {"segmentation": seg, "iscrowd": crowd}}
    originals = [orig(1, 0, [lshape], [0, 0, 100, 100]), orig(2, 1, rle, [50, 50, 100, 100])]
    converted = [{"category_id": 1, "kind": "poly", "poly": lshape, "box": [0, 0, 100, 100]},
                 {"category_id": 1, "kind": "poly", "poly": [60, 60, 80, 60, 80, 80, 60, 80],
                  "box": [60, 60, 80, 80]}]
    rows, extra = measure_image(originals, converted, "seg", h, w)
    by = {r["ann_id"]: r for r in rows}
    assert by[1]["n_fragments"] == 0 and not by[1]["material"]
    assert by[2]["status"] == "crowd_kept"
    assert extra["n_added"] == 0


def test_split_and_box_fallback_are_detected():
    table = object_table(make_fixture())[1]
    parts = [[300, 100, 360, 100, 360, 160, 300, 160], [380, 100, 440, 100, 440, 160, 380, 160]]
    conv = [
        {"category_id": 3, "kind": "poly", "poly": parts[0], "box": [300, 100, 360, 160]},
        {"category_id": 3, "kind": "poly", "poly": parts[1], "box": [380, 100, 440, 160]},
        {"category_id": 17, "kind": "poly", "poly": [450, 300, 530, 300, 530, 380, 450, 380],
         "box": [450, 300, 530, 380]},  # ring replaced by its box
    ]
    rows, extra, _ = _rows(1, conv, "seg")
    # one object written as two: one match plus one fragment
    assert rows[2]["n_fragments"] == 1 and rows[2]["material"]
    assert rows[2]["mask_iou_union"] > 0.99
    assert rows[3]["rect_fallback"] and rows[3]["mask_iou"] < 0.9
    assert extra["n_added"] == 0


def test_rle_variant_keeps_pixels():
    coco = make_fixture()
    rle = make_rle_variant(coco)
    size = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
    for a, b in zip(coco["annotations"], rle["annotations"]):
        h, w = size[a["image_id"]]
        if isinstance(a["segmentation"], list) and a["segmentation"]:
            assert isinstance(b["segmentation"], dict)
            assert mask_iou(ann_to_mask(a, h, w), ann_to_mask(b, h, w)) == 1.0


def test_yolo_parser(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("2 0.5 0.5 0.25 0.5\n1 0.1 0.1 0.2 0.1 0.2 0.2\n3 0.5 0.5\nfoo bar\n")
    objs, bad = parse_yolo_txt(p, 200, 100)
    assert bad == 2
    assert objs[0]["box"] == [75.0, 25.0, 125.0, 75.0]
    assert objs[1]["kind"] == "poly" and np.allclose(objs[1]["poly"], [20, 10, 40, 10, 40, 20])


def test_voc_conventions(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text("<annotation><object><name>car</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
                 "<xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>")
    assert parse_voc_xml(p, "devkit")[0][0]["box"] == [0.0, 0.0, 10.0, 20.0]
    assert parse_voc_xml(p, "raw")[0][0]["box"] == [1.0, 1.0, 10.0, 20.0]


def test_end_to_end_on_fixture(tmp_path):
    coco_path = tmp_path / "fixture.json"
    coco_path.write_text(json.dumps(make_fixture()))
    out = tmp_path / "res"
    subprocess.run([sys.executable, str(ROOT / "run_pilot.py"), "--coco", str(coco_path),
                    "--out", str(out)], check=True, cwd=ROOT)
    s = pd.read_csv(out / "summary.csv")
    assert len(s) == 22  # 11 paths x 2 variants
    m = json.loads((out / "manifest.json").read_text())
    assert not any("error" in v for v in m["paths"].values())
    assert (out / "summary.md").exists() and (out / "manifest.json").exists()


def test_fiftyone_paths_on_fixture(tmp_path):
    """FiftyOne writes exact YOLO boxes, keeps crowd regions as ordinary boxes, and
    flags them only with its own <iscrowd> tag in VOC."""
    from convaudit.adapters import run_path
    coco = make_fixture()
    res, _ = run_path("fiftyone_yolo_det", coco, tmp_path / "y", None)
    objs = res["img1.jpg"][0]
    assert len(objs) == 4  # three objects plus the crowd region
    assert sorted(o["category_id"] for o in objs) == [1, 1, 3, 17]
    res, _ = run_path("fiftyone_voc", coco, tmp_path / "v", None)
    flags = sorted(o["attrs"].get("is_crowd") for o in res["img1.jpg"][0])
    assert flags == ["0", "0", "0", "1"]
