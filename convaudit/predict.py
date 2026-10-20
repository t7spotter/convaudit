"""Run fixed, pretrained detectors once and save COCO-format predictions.

Stage 1 holds predictions fixed and changes only the ground truth, so these
are computed once per model. Output is written image by image to a .jsonl file,
so an interrupted run resumes where it stopped.

Models (weights download automatically on first use):
  frcnn        torchvision Faster R-CNN ResNet-50 FPN v2, COCO weights (boxes)
  mrcnn        torchvision Mask R-CNN ResNet-50 FPN v2, COCO weights (boxes, masks)
  yolo11s      Ultralytics YOLO11s (boxes)
  yolo11s-seg  Ultralytics YOLO11s-seg (boxes, masks)
  rtdetr-l     Ultralytics RT-DETR-L (boxes)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from .coco_io import COCO80_IDS

MODELS = ("frcnn", "mrcnn", "yolo11s", "yolo11s-seg", "rtdetr-l")
OUTPUTS = {"frcnn": ("bbox",), "mrcnn": ("bbox", "segm"), "yolo11s": ("bbox",),
           "yolo11s-seg": ("bbox", "segm"), "rtdetr-l": ("bbox",),
           "oracle": ("bbox", "segm")}  # test-only stand-in: returns the original labels


def _rle(mask: np.ndarray) -> dict:
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def _device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Torchvision:
    def __init__(self, name: str, device: str):
        import torch
        from torchvision.models import detection as D
        if name == "frcnn":
            w = D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            self.model = D.fasterrcnn_resnet50_fpn_v2(weights=w, box_detections_per_img=100)
        else:
            w = D.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            self.model = D.maskrcnn_resnet50_fpn_v2(weights=w, box_detections_per_img=100)
        self.tf = w.transforms()
        self.torch = torch
        self.device = device
        self.model.eval().to(device)
        self.masks = name == "mrcnn"

    def __call__(self, path: Path, image_id: int):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        x = self.tf(img).to(self.device)
        with self.torch.inference_mode():
            out = self.model([x])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()  # torchvision COCO labels are category ids
        res = {"bbox": [], "segm": []}
        masks = (out["masks"][:, 0] > 0.5).cpu().numpy() if self.masks else None
        for i in range(len(scores)):
            x1, y1, x2, y2 = map(float, boxes[i])
            d = {"image_id": image_id, "category_id": int(labels[i]), "score": float(scores[i])}
            res["bbox"].append({**d, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            if masks is not None:
                res["segm"].append({**d, "segmentation": _rle(masks[i])})
        return res


class _Ultralytics:
    def __init__(self, name: str, device: str):
        from ultralytics import RTDETR, YOLO
        self.model = RTDETR(f"{name}.pt") if name.startswith("rtdetr") else YOLO(f"{name}.pt")
        self.device = device
        self.masks = name.endswith("-seg")

    def __call__(self, path: Path, image_id: int):
        r = self.model.predict(str(path), conf=0.001, iou=0.7, max_det=100, device=self.device,
                               retina_masks=True, verbose=False)[0]
        res = {"bbox": [], "segm": []}
        if r.boxes is None or len(r.boxes) == 0:
            return res
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy() > 0.5 if (self.masks and r.masks is not None) else None
        for i in range(len(conf)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            d = {"image_id": image_id, "category_id": COCO80_IDS[cls[i]], "score": float(conf[i])}
            res["bbox"].append({**d, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            if masks is not None:
                res["segm"].append({**d, "segmentation": _rle(masks[i])})
        return res


class _Oracle:
    """Test stand-in: predicts every non-crowd original object with score 1."""

    def __init__(self, coco: dict):
        self.by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            if not a.get("iscrowd"):
                self.by_img.setdefault(a["image_id"], []).append(a)
        self.size = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}

    def __call__(self, path: Path, image_id: int):
        from .geometry import ann_to_mask
        h, w = self.size[image_id]
        res = {"bbox": [], "segm": []}
        for a in self.by_img.get(image_id, []):
            d = {"image_id": image_id, "category_id": a["category_id"], "score": 1.0}
            res["bbox"].append({**d, "bbox": list(a["bbox"])})
            if a.get("segmentation"):
                res["segm"].append({**d, "segmentation": _rle(ann_to_mask(a, h, w))})
        return res


def _load(name: str, device: str, coco: dict | None = None):
    if name == "oracle":
        return _Oracle(coco)
    return _Torchvision(name, device) if name in ("frcnn", "mrcnn") else _Ultralytics(name, device)


def predict(name: str, coco: dict, images_dir: Path, out_dir: Path, device: str = "auto") -> dict:
    """Run one model on every image in `coco`; return paths of the final JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = {t: out_dir / f"{name}.{t}.json" for t in OUTPUTS[name]}
    images = sorted(coco["images"], key=lambda im: im["id"])
    ids = {im["id"] for im in images}
    meta = out_dir / f"{name}.images.json"
    if all(p.exists() for p in final.values()) and meta.exists() \
            and set(json.loads(meta.read_text())) == ids:
        return final  # already computed for exactly these images
    # Per-image results are kept across runs (a quick run's images are reused).
    part = out_dir / f"{name}.partial.jsonl"
    done = set()
    if part.exists():
        with open(part, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["image_id"])
    todo = [im for im in images if im["id"] not in done]
    done &= ids
    dev = _device(device)
    model = _load(name, dev, coco)
    t0, n = time.time(), 0
    with open(part, "a", encoding="utf-8") as f:
        for im in todo:
            p = images_dir / im["file_name"]
            try:
                res = model(p, im["id"])
            except (RuntimeError, NotImplementedError) as e:
                if dev == "cpu":
                    raise
                print(f"   {name}: {dev} failed ({type(e).__name__}); switching to cpu", flush=True)
                dev = "cpu"
                model = _load(name, dev, coco)
                res = model(p, im["id"])
            f.write(json.dumps({"image_id": im["id"], **res}) + "\n")
            n += 1
            if n % 250 == 0:
                rate = (time.time() - t0) / n
                print(f"   {name}: {len(done) + n}/{len(images)} images, "
                      f"~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)
    merged = {t: [] for t in OUTPUTS[name]}
    with open(part, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["image_id"] not in ids:
                continue
            for t in merged:
                merged[t] += r.get(t, [])
    for t, p in final.items():
        p.write_text(json.dumps(merged[t]), encoding="utf-8")
    meta.write_text(json.dumps(sorted(ids)), encoding="utf-8")
    return final
