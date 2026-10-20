"""Stage 2: does training on converted labels change accuracy on the original labels?

One fixed subset of COCO train2017 is converted by each tool. The same model,
initialization, hyperparameters and seeds are trained on each label set, and
every model is evaluated against the original COCO val2017 ground truth.
Only the training labels differ between conditions.

Layout of the data folder (built by `prepare`):
  coco/train_subset.json      original annotations of the sampled images
  coco/val2017.json           original validation annotations (evaluation truth)
  images/train/               the sampled training images
  images/val/                 50 val images; Ultralytics requires a val split
  labelsets/<key>/{train,val} YOLO label files exactly as each tool wrote them
  labelsets/<key>/names.json  class names in the tool's index order
  audit/                      what each tool changed in the training subset
  labels/                     the label set of the run in progress (swapped per run)
"""

from __future__ import annotations

import json
import random
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import run_path
from .coco_io import COCO80_IDS, load_coco, make_rle_variant, object_table, shards, write_coco
from .measure import measure_image
from .report import summarize

# label set key -> (variant, conversion path)
LABELSETS = {
    "reference_det": ("original", "ultralytics_yolo_det"),
    "reference_seg": ("original", "ultralytics_yolo_seg"),
    "supervision": ("original", "supervision_yolo"),
    "datumaro": ("original", "datumaro_yolo_det"),
    "ultralytics_rle": ("rle", "ultralytics_yolo_seg"),
}
# condition -> (task, label set). The reference conditions use Ultralytics'
# converter, which the audit found to preserve COCO polygons (crowd regions are
# dropped, which is also standard COCO training practice).
CONDITIONS = {
    "det_reference": ("detect", "reference_det"),
    "det_reference_poly": ("detect", "reference_seg"),
    "det_supervision": ("detect", "supervision"),
    "det_datumaro": ("detect", "datumaro"),
    "seg_reference": ("segment", "reference_seg"),
    "seg_supervision": ("segment", "supervision"),
    "seg_ultralytics_rle": ("segment", "ultralytics_rle"),
}
# Each condition is compared with the reference that shares its label *form*.
# Ultralytics augments polygon labels differently from box labels (it re-derives
# boxes from the transformed polygons), so a detector trained on supervision's
# polygon files is compared with one trained on Ultralytics' polygon files, and
# Datumaro's box files with Ultralytics' box files. Protocol amendment of
# 2026-10-15, made after the first two runs started and before any comparison
# was computed (see README).
REFERENCE_OF = {
    "det_reference_poly": None, "det_reference": None, "seg_reference": None,
    "det_supervision": "det_reference_poly",
    "det_datumaro": "det_reference",
    "seg_supervision": "seg_reference",
    "seg_ultralytics_rle": "seg_reference",
}
REFERENCE = {"detect": "det_reference", "segment": "seg_reference"}  # kept for older callers
SEEDS = (0, 1, 2)
# Pre-specified training recipe, identical for every condition.
HYP = dict(epochs=30, imgsz=640, batch=16, optimizer="SGD", lr0=0.01, momentum=0.937,
           weight_decay=0.0005, warmup_epochs=3.0, close_mosaic=10, cos_lr=False, amp=True,
           cache=False)
N_VAL_SPLIT = 50


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, tries: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def _fetch_images(coco: dict, dest: Path, source_dir: Path | None, threads: int = 16) -> None:
    def one(im):
        target = dest / im["file_name"]
        if target.exists() and target.stat().st_size > 0:
            return
        if source_dir is not None and (source_dir / im["file_name"]).exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / im["file_name"], target)
        else:
            _download(im["coco_url"], target)
    with ThreadPoolExecutor(threads) as ex:
        for i, _ in enumerate(ex.map(one, coco["images"]), 1):
            if i % 1000 == 0:
                print(f"   images: {i}/{len(coco['images'])}", flush=True)


def _subset(coco: dict, ids: set) -> dict:
    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = [im for im in coco["images"] if im["id"] in ids]
    out["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
    return out


def _tool_label_file(path: str, out: Path, file_name: str) -> Path:
    """Where each tool wrote the label file of one image (see adapters.py)."""
    stem_txt = Path(file_name).with_suffix(".txt")
    if path.startswith("ultralytics"):
        return out / "labels" / "shard" / stem_txt
    if path == "supervision_yolo":
        return out / "labels" / (Path(file_name).stem + ".txt")
    if path == "datumaro_yolo_det":
        return out / "obj_shard_data" / stem_txt
    raise ValueError(path)


def _tool_names(path: str, out: Path, coco: dict, meta: dict) -> list[str]:
    """Class names in the index order the tool used, read from the tool's own files."""
    from .parsers import read_names_file, read_yaml_names
    if path.startswith("ultralytics"):
        by_id = {c["id"]: c["name"] for c in coco["categories"]}
        if meta.get("cls91to80"):
            return [by_id.get(i, f"unused_{i}") for i in COCO80_IDS]
        return [by_id.get(i + 1, f"unused_{i + 1}") for i in range(max(by_id))]
    if path == "supervision_yolo":
        return read_yaml_names(out / "data.yaml")
    if path == "datumaro_yolo_det":
        return read_names_file(out / "obj.names")
    raise ValueError(path)


def build_labelset(key: str, coco_by_variant: dict, split: str, images_dir: Path,
                   data_dir: Path, shard_size: int = 250) -> pd.DataFrame:
    """Convert one split with one tool; copy the tool's label files; audit them."""
    variant, path = LABELSETS[key]
    coco = coco_by_variant[variant]
    dest = data_dir / "labelsets" / key / split
    dest.mkdir(parents=True, exist_ok=True)
    table = object_table(coco)
    rows, extras, names = [], [], None
    for k, shard in enumerate(shards(coco, shard_size)):
        work = data_dir / "work" / key / split / f"shard{k:04d}"
        result, meta = run_path(path, shard, work, images_dir)
        out = work / "out"
        shard_names = _tool_names(path, out, shard, meta)
        if names is None:
            names = shard_names
        elif names != shard_names:
            raise RuntimeError(f"{key}: class order differs between shards")
        for im in shard["images"]:
            src = _tool_label_file(path, out, im["file_name"])
            target = dest / (Path(im["file_name"]).stem + ".txt")
            if src.exists():
                shutil.copy2(src, target)
            else:
                target.write_text("")  # no objects written: a background image
            objs, bad = result[im["file_name"]]
            r, extra = measure_image(table[im["id"]], objs, meta["kind"], im["height"], im["width"])
            for x in r:
                x.update(variant=key, path=path)
            rows += r
            extras.append({"variant": key, "path": path, "image_id": im["id"],
                           "n_added": extra["n_added"], "n_malformed": bad})
        shutil.rmtree(work)
    (data_dir / "labelsets" / key / "names.json").write_text(json.dumps(names), encoding="utf-8")
    if split != "train":
        return pd.DataFrame()
    return summarize(pd.DataFrame(rows), pd.DataFrame(extras))


def prepare(train_ann: Path, val_ann: Path, data_dir: Path, n_train: int = 10000, seed: int = 0,
            train_images: Path | None = None, val_images: Path | None = None,
            labelsets: list[str] | None = None) -> None:
    """Sample the training subset, fetch its images and build every label set."""
    data_dir.mkdir(parents=True, exist_ok=True)
    labelsets = labelsets or list(LABELSETS)
    print("[prepare] loading annotations ...", flush=True)
    full = load_coco(train_ann)
    ids = sorted(im["id"] for im in full["images"])
    chosen = set(random.Random(seed).sample(ids, min(n_train, len(ids))))
    train = _subset(full, chosen)
    del full
    val = load_coco(val_ann)
    val_ids = sorted(im["id"] for im in val["images"])
    val_small = _subset(val, set(random.Random(seed).sample(val_ids, min(N_VAL_SPLIT, len(val_ids)))))
    write_coco(train, data_dir / "coco" / "train_subset.json")
    write_coco(val, data_dir / "coco" / "val2017.json")

    print(f"[prepare] fetching {len(train['images'])} training images ...", flush=True)
    _fetch_images(train, data_dir / "images" / "train", train_images)
    _fetch_images(val_small, data_dir / "images" / "val", val_images)

    variants = {"original": {"train": train, "val": val_small}}
    if any(LABELSETS[k][0] == "rle" for k in labelsets):
        print("[prepare] building the RLE-stored variant ...", flush=True)
        variants["rle"] = {"train": make_rle_variant(train), "val": make_rle_variant(val_small)}
    audits = []
    for key in labelsets:
        print(f"[prepare] label set {key} ...", flush=True)
        for split in ("train", "val"):
            by_variant = {v: d[split] for v, d in variants.items()}
            s = build_labelset(key, by_variant, split, data_dir / "images" / split, data_dir)
            if split == "train":
                audits.append(s)
    (data_dir / "audit").mkdir(exist_ok=True)
    pd.concat(audits).to_csv(data_dir / "audit" / "train_subset_audit.csv", index=False)
    meta = {"n_train": len(train["images"]), "n_train_annotations": len(train["annotations"]),
            "seed": seed, "train_image_ids": sorted(chosen), "val_split_ids": sorted(im["id"] for im in val_small["images"]),
            "labelsets": {k: LABELSETS[k] for k in labelsets},
            "tools": {p: version(p) for p in ("ultralytics", "supervision", "datumaro", "pycocotools")}}
    (data_dir / "prepare.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(data_dir / "work", ignore_errors=True)
    print("[prepare] done", flush=True)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _use_labelset(data_dir: Path, key: str) -> Path:
    """Put one label set in place for training and write its dataset file."""
    labels = data_dir / "labels"
    if labels.exists():
        shutil.rmtree(labels)
    shutil.copytree(data_dir / "labelsets" / key, labels, ignore=shutil.ignore_patterns("names.json"))
    for c in data_dir.glob("**/*.cache"):
        c.unlink()
    names = json.loads((data_dir / "labelsets" / key / "names.json").read_text())
    yaml_path = data_dir / f"data_{key}.yaml"
    yaml_path.write_text(
        f"path: {data_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(names)), encoding="utf-8")
    return yaml_path


def _gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _checkpoint_epoch(path: Path) -> int:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1)) if ckpt.get("optimizer") is not None else -1


def run_name(condition: str, seed: int) -> str:
    return f"{condition}_s{seed}"


def train_run(data_dir: Path, project: Path, condition: str, seed: int, val_images: Path,
              device: str | None = None, workers: int = 2, pretrained: bool = True,
              hyp_overrides: dict | None = None) -> Path:
    """Train one condition with one seed (resuming if interrupted), then predict val2017."""
    from ultralytics import YOLO
    task, key = CONDITIONS[condition]
    name = run_name(condition, seed)
    run_dir = project / name
    done = run_dir / "done.json"
    if done.exists():
        return run_dir
    yaml_path = _use_labelset(data_dir, key)
    hyp = {**HYP, **(hyp_overrides or {})}
    last = run_dir / "weights" / "last.pt"
    t0 = time.time()
    resumes = run_dir / "resumes.json"  # every resume is logged for the paper
    if last.exists() and not (run_dir / "weights" / "trained.flag").exists():
        ep = _checkpoint_epoch(last)
        if ep >= 0:  # unfinished: continue where it stopped
            print(f"[train] resuming {name} after epoch {ep + 1}", flush=True)
            log = json.loads(resumes.read_text()) if resumes.exists() else []
            log.append({"after_epoch": ep + 1, "gpu": _gpu_name(), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            resumes.write_text(json.dumps(log, indent=1))
            YOLO(str(last)).train(resume=True)
        # epoch -1 means training finished (optimizer stripped); go on to prediction
    elif not last.exists():
        print(f"[train] {name}", flush=True)
        model = YOLO("yolo11n-seg.yaml" if task == "segment" else "yolo11n.yaml")
        if pretrained:
            # ImageNet-pretrained backbone; Ultralytics transfers layers matching
            # by name and shape. Do not pass pretrained=False to train(): in this
            # Ultralytics version that discards the loaded weights.
            model.load("yolo11n-cls.pt")
        model.train(data=str(yaml_path), project=str(project), name=name, exist_ok=True, seed=seed,
                    deterministic=True, val=False, plots=False, workers=workers, device=device, **hyp)
    (run_dir / "weights" / "trained.flag").write_text("ok")
    names = json.loads((data_dir / "labelsets" / key / "names.json").read_text())
    val = load_coco(data_dir / "coco" / "val2017.json")
    preds = predict_val(run_dir / "weights" / "last.pt", names, val, val_images, run_dir, device,
                        masks=task == "segment", imgsz=hyp["imgsz"])
    done.write_text(json.dumps({
        "condition": condition, "task": task, "labelset": key, "seed": seed, "hyp": hyp,
        "pretrained": "yolo11n-cls.pt" if pretrained else None, "minutes": round((time.time() - t0) / 60, 1),
        "gpu_at_finish": _gpu_name(),
        "resumes": json.loads(resumes.read_text()) if resumes.exists() else [],
        "predictions": {k: str(v) for k, v in preds.items()},
        "tools": {p: version(p) for p in ("ultralytics", "torch")}}, indent=1), encoding="utf-8")
    return run_dir


def predict_val(weights: Path, names: list[str], val: dict, images_dir: Path, run_dir: Path,
                device: str | None, masks: bool, imgsz: int = 640) -> dict:
    """COCO-format predictions on every val2017 image, classes mapped by name."""
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO
    name_to_cat = {c["name"]: c["id"] for c in val["categories"]}
    idx_to_cat = [name_to_cat.get(n) for n in names]
    model = YOLO(str(weights))
    out = {"bbox": []}
    if masks:
        out["segm"] = []
    for k, im in enumerate(sorted(val["images"], key=lambda x: x["id"]), 1):
        r = model.predict(str(images_dir / im["file_name"]), conf=0.001, iou=0.7, max_det=100,
                          imgsz=imgsz, device=device, retina_masks=True, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        mk = r.masks.data.cpu().numpy() > 0.5 if (masks and r.masks is not None) else None
        for i in range(len(conf)):
            cat = idx_to_cat[cls[i]] if cls[i] < len(idx_to_cat) else None
            if cat is None:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            d = {"image_id": im["id"], "category_id": cat, "score": float(conf[i])}
            out["bbox"].append({**d, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            if mk is not None:
                rle = mask_utils.encode(np.asfortranarray(mk[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                out["segm"].append({**d, "segmentation": rle})
        if k % 1000 == 0:
            print(f"   predictions: {k}/{len(val['images'])}", flush=True)
    files = {}
    for t, v in out.items():
        files[t] = run_dir / f"predictions_val2017.{t}.json"
        files[t].write_text(json.dumps(v), encoding="utf-8")
    return files


def queue(data_dir: Path, project: Path, val_images: Path, conditions=None, seeds=SEEDS, **kw) -> None:
    """Train every pending run in a fixed order; safe to call again after an interruption."""
    conditions = conditions or list(CONDITIONS)
    # seed-major order: every condition gets seed 0 before any gets seed 1,
    # so a partial budget still gives a complete first comparison
    for seed in seeds:
        for c in conditions:
            train_run(data_dir, project, c, seed, val_images, **kw)


def status(project: Path, conditions=None, seeds=SEEDS) -> pd.DataFrame:
    conditions = conditions or list(CONDITIONS)
    rows = []
    for seed in seeds:
        for c in conditions:
            d = project / run_name(c, seed)
            st = "done" if (d / "done.json").exists() else (
                "predicting" if (d / "weights" / "trained.flag").exists() else
                "training" if (d / "weights" / "last.pt").exists() else "pending")
            rows.append({"run": run_name(c, seed), "status": st})
    return pd.DataFrame(rows)
