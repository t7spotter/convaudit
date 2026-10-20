"""Adapters: run one third-party conversion path and read its output back.

Each adapter converts one shard (a COCO instances file) and returns, for every
image file_name, the objects found in the tool's output with the class index
resolved to a COCO category_id. Reading back always uses convaudit.parsers.

Adding a tool means adding one function here and one entry in PATHS.
"""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .coco_io import COCO80_IDS, write_coco, write_stub_images
from .parsers import (parse_voc_xml, parse_yolo_txt, read_names_file,
                      read_yaml_names)


def tool_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def _name_to_cat(coco: dict) -> dict[str, int]:
    names = [c["name"] for c in coco["categories"]]
    if len(names) != len(set(names)):
        raise ValueError("category names are not unique; name-based mapping is ambiguous")
    return {c["name"]: c["id"] for c in coco["categories"]}


def _resolve(objs: list[dict], index_to_cat) -> list[dict]:
    for o in objs:
        o["category_id"] = index_to_cat(o)
    return objs


def _quiet():
    """Silence tool progress bars and logs while converting."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


# --------------------------------------------------------------------------- #
# Ultralytics: ultralytics.data.converter.convert_coco
# --------------------------------------------------------------------------- #
def _ultralytics(shard: dict, work: Path, use_segments: bool):
    from ultralytics.data.converter import convert_coco
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    ann_dir = work / "in"
    write_coco(shard, ann_dir / "instances_shard.json")
    out = work / "out"  # must not exist: convert_coco would rename it
    cat_ids = {c["id"] for c in shard["categories"]}
    cls91to80 = cat_ids <= set(COCO80_IDS)
    with _quiet():
        convert_coco(labels_dir=str(ann_dir), save_dir=str(out),
                     use_segments=use_segments, cls91to80=cls91to80)
    # Documented convention: with cls91to80 the index is the COCO-80 position,
    # otherwise it is category_id - 1.
    def to_cat(o):
        if cls91to80:
            return COCO80_IDS[o["cls"]] if 0 <= o["cls"] < 80 else None
        return o["cls"] + 1

    result = {}
    for im in shard["images"]:
        p = (out / "labels" / "shard" / im["file_name"]).with_suffix(".txt")
        objs, bad = parse_yolo_txt(p, im["width"], im["height"])
        result[im["file_name"]] = (_resolve(objs, to_cat), bad)
    return result, {"cls91to80": cls91to80}


def ultralytics_yolo_det(shard, work, images_dir):
    return _ultralytics(shard, work, use_segments=False)


def ultralytics_yolo_seg(shard, work, images_dir):
    return _ultralytics(shard, work, use_segments=True)


# --------------------------------------------------------------------------- #
# Roboflow supervision: DetectionDataset.from_coco -> as_yolo / as_pascal_voc
# --------------------------------------------------------------------------- #
def _supervision_load(shard, work, images_dir):
    import supervision as sv
    json_path = work / "in" / "shard.json"
    write_coco(shard, json_path)
    if images_dir is None:  # the tool opens every image to read its size
        images_dir = work / "stub_images"
        write_stub_images(shard, images_dir)
    with _quiet():
        ds = sv.DetectionDataset.from_coco(images_directory_path=str(images_dir),
                                           annotations_path=str(json_path))
    return ds


def supervision_yolo(shard, work, images_dir):
    ds = _supervision_load(shard, work, images_dir)
    out = work / "out"
    with _quiet():
        ds.as_yolo(annotations_directory_path=str(out / "labels"),
                   data_yaml_path=str(out / "data.yaml"))
    names = read_yaml_names(out / "data.yaml")
    n2c = _name_to_cat(shard)

    def to_cat(o):
        return n2c.get(names[o["cls"]]) if 0 <= o["cls"] < len(names) else None

    result = {}
    for im in shard["images"]:
        p = out / "labels" / (Path(im["file_name"]).stem + ".txt")
        objs, bad = parse_yolo_txt(p, im["width"], im["height"])
        result[im["file_name"]] = (_resolve(objs, to_cat), bad)
    return result, {"masks_present": True}


def supervision_voc(shard, work, images_dir, convention="devkit"):
    ds = _supervision_load(shard, work, images_dir)
    out = work / "out"
    with _quiet():
        ds.as_pascal_voc(annotations_directory_path=str(out / "Annotations"))
    n2c = _name_to_cat(shard)
    result = {}
    for im in shard["images"]:
        p = out / "Annotations" / (Path(im["file_name"]).stem + ".xml")
        objs, bad = parse_voc_xml(p, convention)
        result[im["file_name"]] = (_resolve(objs, lambda o: n2c.get(o["name"])), bad)
    return result, {"voc_reading": convention}


# --------------------------------------------------------------------------- #
# Datumaro (the CVAT team's dataset library): import coco -> export yolo / voc
# --------------------------------------------------------------------------- #
def _datumaro_load(shard, work):
    import datumaro as dm
    root = work / "in"
    write_coco(shard, root / "annotations" / "instances_shard.json")
    (root / "images" / "shard").mkdir(parents=True, exist_ok=True)  # required layout
    with _quiet():
        return dm.Dataset.import_from(str(root), "coco_instances")


def datumaro_yolo_det(shard, work, images_dir):
    ds = _datumaro_load(shard, work)
    out = work / "out"
    with _quiet():
        ds.export(str(out), "yolo", save_media=False)
    names = read_names_file(out / "obj.names")
    n2c = _name_to_cat(shard)

    def to_cat(o):
        return n2c.get(names[o["cls"]]) if 0 <= o["cls"] < len(names) else None

    result = {}
    for im in shard["images"]:
        p = (out / "obj_shard_data" / im["file_name"]).with_suffix(".txt")
        objs, bad = parse_yolo_txt(p, im["width"], im["height"])
        result[im["file_name"]] = (_resolve(objs, to_cat), bad)
    return result, {}


def datumaro_voc(shard, work, images_dir, convention="devkit"):
    ds = _datumaro_load(shard, work)
    out = work / "out"
    with _quiet():
        ds.export(str(out), "voc_detection", save_media=False)
    n2c = _name_to_cat(shard)
    result = {}
    for im in shard["images"]:
        p = (out / "Annotations" / im["file_name"]).with_suffix(".xml")
        objs, bad = parse_voc_xml(p, convention)
        result[im["file_name"]] = (_resolve(objs, lambda o: n2c.get(o["name"])), bad)
    return result, {"voc_reading": convention}


# --------------------------------------------------------------------------- #
# FiftyOne (Voxel51): Dataset.from_dir(COCODetectionDataset) -> export
# --------------------------------------------------------------------------- #
def _fiftyone_export(shard, work, images_dir, dataset_type: str):
    import fiftyone as fo
    json_path = work / "in" / "shard.json"
    write_coco(shard, json_path)
    if images_dir is None:  # sample paths must point at files; pixels are not read
        images_dir = work / "stub_images"
        write_stub_images(shard, images_dir)
    # Class list in category-id order, passed explicitly as FiftyOne's docs advise;
    # without it the index order follows the labels' first appearance.
    names = [c["name"] for c in sorted(shard["categories"], key=lambda c: c["id"])]
    out = work / "out"
    with _quiet():
        ds = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset,
                                 data_path=str(images_dir), labels_path=str(json_path),
                                 label_types="detections", label_field="ground_truth")
        try:
            extra = {"classes": names} if dataset_type == "YOLOv5Dataset" else {}
            ds.export(export_dir=str(out), dataset_type=getattr(fo.types, dataset_type),
                      label_field="ground_truth", export_media=False, **extra)
        finally:
            ds.delete()
    return out, names


def fiftyone_yolo_det(shard, work, images_dir):
    out, names = _fiftyone_export(shard, work, images_dir, "YOLOv5Dataset")
    n2c = _name_to_cat(shard)

    def to_cat(o):
        return n2c.get(names[o["cls"]]) if 0 <= o["cls"] < len(names) else None

    result = {}
    for im in shard["images"]:
        p = out / "labels" / "val" / (Path(im["file_name"]).stem + ".txt")
        objs, bad = parse_yolo_txt(p, im["width"], im["height"])
        result[im["file_name"]] = (_resolve(objs, to_cat), bad)
    return result, {}


def fiftyone_voc(shard, work, images_dir, convention="devkit"):
    out, _ = _fiftyone_export(shard, work, images_dir, "VOCDetectionDataset")
    n2c = _name_to_cat(shard)
    result = {}
    for im in shard["images"]:
        p = out / "labels" / (Path(im["file_name"]).stem + ".xml")
        objs, bad = parse_voc_xml(p, convention)
        result[im["file_name"]] = (_resolve(objs, lambda o: n2c.get(o["name"])), bad)
    return result, {"voc_reading": convention}


# path name -> (function, output kind, package to report the version of).
# VOC output is read twice: with the VOC devkit's 1-based convention (how
# standard loaders read it) and as written, because tools disagree on it.
PATHS = {
    "ultralytics_yolo_det": (ultralytics_yolo_det, "det", "ultralytics"),
    "ultralytics_yolo_seg": (ultralytics_yolo_seg, "seg", "ultralytics"),
    "supervision_yolo": (supervision_yolo, "seg", "supervision"),
    "supervision_voc": (supervision_voc, "det", "supervision"),
    "supervision_voc_raw": (partial(supervision_voc, convention="raw"), "det", "supervision"),
    "datumaro_yolo_det": (datumaro_yolo_det, "det", "datumaro"),
    "datumaro_voc": (datumaro_voc, "det", "datumaro"),
    "datumaro_voc_raw": (partial(datumaro_voc, convention="raw"), "det", "datumaro"),
    # FiftyOne's YOLO export writes boxes only (masks are not turned into polygons),
    # so it has no segmentation path.
    "fiftyone_yolo_det": (fiftyone_yolo_det, "det", "fiftyone"),
    "fiftyone_voc": (fiftyone_voc, "det", "fiftyone"),
    "fiftyone_voc_raw": (partial(fiftyone_voc, convention="raw"), "det", "fiftyone"),
}


def run_path(name: str, shard: dict, work: Path, images_dir: Path | None):
    fn, kind, pkg = PATHS[name]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    result, meta = fn(shard, work, images_dir)
    return result, {"kind": kind, "package": pkg, "version": tool_version(pkg), **meta}
