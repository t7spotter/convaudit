"""Stage 1 checks: AP equals pycocotools; weights equal duplicated images; GT builder."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from convaudit.ap import APTable, bootstrap_weights
from convaudit.coco_io import object_table
from convaudit.gt_build import compose
from convaudit.measure import measure_image

sys.path.insert(0, str(Path(__file__).parent))
from fixture import make_fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(n_img=40, seed=1):
    """Small GT with polygons and crowd, plus noisy predictions with false positives."""
    rng = np.random.default_rng(seed)
    images, anns, preds_b, preds_s, aid = [], [], [], [], 1
    for i in range(1, n_img + 1):
        w, h = 320, 240
        images.append({"id": i, "file_name": f"{i}.jpg", "width": w, "height": h})
        for _ in range(rng.integers(0, 6)):
            s = float(rng.choice([8, 20, 60, 110]))
            x, y = rng.uniform(0, w - s), rng.uniform(0, h - s)
            poly = [x, y, x + s, y, x + s, y + s * 0.8, x + s * 0.2, y + s]
            cat = int(rng.choice([1, 3, 17]))
            crowd = int(rng.random() < 0.05)
            r = mask_utils.merge(mask_utils.frPyObjects([poly], h, w))
            bb = mask_utils.toBbox(r).tolist()
            seg = [poly]
            if crowd:
                seg = mask_utils.frPyObjects([poly], h, w)[0]
                seg["counts"] = seg["counts"].decode()
            anns.append({"id": aid, "image_id": i, "category_id": cat, "segmentation": seg,
                         "area": float(mask_utils.area(r)), "bbox": bb, "iscrowd": crowd})
            aid += 1
            for _ in range(rng.integers(1, 3)):  # 1-2 noisy detections per object
                j = rng.normal(0, s * 0.08, 4)
                b = [bb[0] + j[0], bb[1] + j[1], max(1, bb[2] + j[2]), max(1, bb[3] + j[3])]
                c = cat if rng.random() > 0.1 else int(rng.choice([1, 3, 17]))
                sc = float(rng.uniform(0.05, 1))
                preds_b.append({"image_id": i, "category_id": c, "bbox": b, "score": sc})
                m = mask_utils.frPyObjects([[b[0], b[1], b[0] + b[2], b[1], b[0] + b[2], b[1] + b[3],
                                             b[0], b[1] + b[3]]], h, w)[0]
                m["counts"] = m["counts"].decode()
                preds_s.append({"image_id": i, "category_id": c, "segmentation": m, "score": sc})
        for _ in range(rng.integers(0, 3)):  # false positives
            b = [rng.uniform(0, 250), rng.uniform(0, 180), 30.0, 30.0]
            preds_b.append({"image_id": i, "category_id": int(rng.choice([1, 3, 17])), "bbox": b,
                            "score": float(rng.uniform(0, 0.6))})
    cats = [{"id": c, "name": n} for c, n in [(1, "person"), (3, "car"), (17, "cat")]]
    return {"images": images, "annotations": anns, "categories": cats}, preds_b, preds_s


def _official(gt_dict, preds, iou_type):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(gt_dict)
        gt.createIndex()
        dt = gt.loadRes(copy.deepcopy(preds))
        E = COCOeval(gt, dt, iou_type)
        E.params.imgIds = sorted(gt.getImgIds())
        E.evaluate()
        T = APTable(E)
        E.accumulate()
        E.summarize()
    return T, E.stats


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_ap_matches_pycocotools(iou_type):
    gt, pb, ps = _synthetic()
    T, stats = _official(gt, pb if iou_type == "bbox" else ps, iou_type)
    ours = [T.ap(), T.ap(iou=0.5), T.ap(iou=0.75), T.ap(area="small"), T.ap(area="medium"), T.ap(area="large")]
    for a, b in zip(ours, stats[:6]):
        if b < 0:  # pycocotools reports -1 when an area range has no objects
            assert np.isnan(a)
        else:
            assert abs(a - 100 * b) < 1e-9


def test_weights_equal_duplicated_images():
    gt, pb, _ = _synthetic(n_img=25, seed=3)
    T, _ = _official(gt, pb, "bbox")
    w = bootstrap_weights(T.n_img, 1, seed=7)[0]
    # build the duplicated dataset explicitly and score it with pycocotools
    dup = {"images": [], "annotations": [], "categories": gt["categories"]}
    dpreds, nid, aid = [], 1, 1
    for idx, img_id in enumerate(T.img_ids):
        for _ in range(int(w[idx])):
            im = dict(next(i for i in gt["images"] if i["id"] == img_id), id=nid)
            dup["images"].append(im)
            for a in gt["annotations"]:
                if a["image_id"] == img_id:
                    dup["annotations"].append(dict(copy.deepcopy(a), id=aid, image_id=nid))
                    aid += 1
            dpreds += [dict(p, image_id=nid) for p in pb if p["image_id"] == img_id]
            nid += 1
    _, stats = _official(dup, dpreds, "bbox")
    assert abs(T.ap(w) - 100 * stats[0]) < 1e-9


def test_full_component_rebuilds_an_exact_copy():
    """With a perfect converter, every component equals the original boxes."""
    coco = make_fixture()
    table = object_table(coco)
    im = coco["images"][0]
    conv = [{"category_id": o["category_id"], "kind": "box", "poly": None, "box": list(o["box"])}
            for o in table[1] if not o["iscrowd"]]
    _, extra = measure_image(table[1], conv, "det", im["height"], im["width"])
    anns, _ = compose("geometry", im, table[1], extra["assign"], with_seg=False)
    got = sorted((a["category_id"], tuple(np.round(a["bbox"], 6))) for a in anns)
    want = sorted((o["category_id"], tuple(np.round(o["ann"]["bbox"], 6))) for o in table[1])
    assert got == want
    full, _ = compose("full", im, table[1], extra["assign"], with_seg=False)
    assert len(full) == 3  # the crowd region is gone: this converter dropped it
    crowd, n = compose("crowd", im, table[1], extra["assign"], with_seg=False)
    assert n == 1 and len(crowd) == 3


def test_stage1_end_to_end(tmp_path):
    """Whole pipeline on the fixture with a stand-in model, no downloads."""
    import cv2
    coco = make_fixture()
    (tmp_path / "imgs").mkdir()
    for im in coco["images"]:
        cv2.imwrite(str(tmp_path / "imgs" / im["file_name"]), np.zeros((im["height"], im["width"], 3), np.uint8))
    (tmp_path / "c.json").write_text(json.dumps(coco))
    out = tmp_path / "s1"
    subprocess.run([sys.executable, str(ROOT / "run_stage1.py"), "--coco", str(tmp_path / "c.json"),
                    "--images", str(tmp_path / "imgs"), "--out", str(out), "--models", "oracle",
                    "--bootstrap", "20", "--workers", "2"], check=True, cwd=ROOT)
    import pandas as pd
    r = pd.read_csv(out / "stage1_results.csv")
    assert r["check_vs_pycocotools"].max() < 1e-6
    # a lossless path keeps AP; a path that splits objects and keeps crowds lowers it
    full = r[(r.component == "full") & (r.iou_type == "bbox") & (r.variant == "original")].set_index("path")
    assert abs(full.loc["ultralytics_yolo_det", "dAP"]) < 0.01
    assert full.loc["supervision_yolo", "dAP"] < 0
    assert (out / "report.md").exists()
    # second run on other paths reuses the saved predictions instead of predicting again
    out2 = tmp_path / "s1_reuse"
    subprocess.run([sys.executable, str(ROOT / "run_stage1.py"), "--coco", str(tmp_path / "c.json"),
                    "--images", str(tmp_path / "imgs"), "--out", str(out2), "--models", "oracle",
                    "--paths", "fiftyone_yolo_det,fiftyone_voc", "--variants", "original",
                    "--predictions-from", str(out / "predictions"),
                    "--bootstrap", "0", "--workers", "1"], check=True, cwd=ROOT)
    assert (out2 / "predictions" / "oracle.bbox.json").read_bytes() == \
        (out / "predictions" / "oracle.bbox.json").read_bytes()
    r2 = pd.read_csv(out2 / "stage1_results.csv")
    assert set(r2.path) == {"fiftyone_yolo_det", "fiftyone_voc"}
