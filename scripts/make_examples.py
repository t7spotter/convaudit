"""Fig. 3 of the paper: real COCO examples of what the converters change.

Run on a machine with COCO val2017 and the pinned tools (Ultralytics 8.4.177, supervision 0.30.9):

    python scripts/make_examples.py --ann annotations/instances_val2017.json --images val2017 --out figures

Only images under the CC BY licence (COCO licence id 4) are used, so the figure can be published.
For each panel the script runs the real converter on a one-image COCO file, reads the output back
and draws it over the source annotation. It prints the image id, file name, Flickr URL and licence
of every image used, for the figure credit. Use --pick panel=N to take the N-th candidate instead
of the first (for example --pick split=2) if an example is not clear enough.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

CC_BY = {4}
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
SRC = "#ffffff"
MM = 1 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})


# ------------------------------------------------------------------ helpers
def one_image_coco(coco, img, anns, folder: Path, name="instances_val2017.json") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(json.dumps({"images": [img], "annotations": anns, "categories": coco["categories"],
                             "licenses": coco.get("licenses", [])}))
    return p


def read_yolo_polys(txt: Path, w: int, h: int):
    polys = []
    for line in txt.read_text().splitlines():
        v = [float(t) for t in line.split()[1:]]
        if len(v) == 4:  # box line: cx cy bw bh
            cx, cy, bw, bh = v
            v = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, cx - bw / 2, cy + bh / 2]
        polys.append(np.array(v).reshape(-1, 2) * [w, h])
    return polys


def src_polys(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return [np.array(s).reshape(-1, 2) for s in seg]
    m = mu.decode(mu.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg)
    import cv2
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(float) for c in cs if len(c) >= 3]


def crop(ax, img, box, margin=0.25, min_size=60):
    """Square window around the box, kept inside the image (narrower only where the image ends)."""
    x, y, bw, bh = box
    W, H = img.size
    side = max(max(bw, bh) * (1 + 2 * margin), min_size, max(bw, bh) + 8)
    sw, sh = min(side, W), min(side, H)
    cx, cy = x + bw / 2, y + bh / 2
    x0 = min(max(cx - sw / 2, -0.5), W - 0.5 - sw)
    y0 = min(max(cy - sh / 2, -0.5), H - 0.5 - sh)
    x1, y1 = x0 + sw, y0 + sh
    ax.imshow(img, interpolation="nearest")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def outline(ax, poly, color, lw=1.4, ls="-", z=3, fill=None):
    ax.add_patch(Polygon(poly, closed=True, fill=fill is not None, fc=fill or "none", ec=color, lw=lw, ls=ls,
                         zorder=z))


def candidates(coco, pred):
    imgs = {i["id"]: i for i in coco["images"] if i.get("license") in CC_BY}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = []
    for a in coco["annotations"]:
        img = imgs.get(a["image_id"])
        if img is not None and pred(a, img):
            out.append((a, img, by_img[a["image_id"]]))
    out.sort(key=lambda t: (len(t[2]), -t[0]["area"]))  # uncluttered images, large objects first
    return out


def credit(tag, img, coco):
    lic = {l["id"]: l for l in coco.get("licenses", [])}.get(img.get("license"), {})
    print(f"[{tag}] image {img['id']} {img['file_name']} {img.get('flickr_url', '')} "
          f"licence: {lic.get('name', img.get('license'))}")


# ------------------------------------------------------------------ panels
def panel_split(ax, coco, images: Path, k: int, tmp: Path):
    import supervision as sv
    c = candidates(coco, lambda a, i: not a["iscrowd"] and isinstance(a["segmentation"], list)
                   and len(a["segmentation"]) >= 2 and 2000 < a["area"] < 60000
                   and min(len(s) for s in a["segmentation"]) >= 12)[k]
    a, img, _ = c
    j = one_image_coco(coco, img, [a], tmp / "split")
    sv.DetectionDataset.from_coco(str(images), str(j), force_masks=True).as_yolo(
        annotations_directory_path=str(tmp / "split" / "yolo"), data_yaml_path=str(tmp / "split" / "d.yaml"))
    polys = read_yolo_polys(next((tmp / "split" / "yolo").glob("*.txt")), img["width"], img["height"])
    im = Image.open(images / img["file_name"]).convert("RGB")
    crop(ax, im, a["bbox"])
    for p in src_polys(a, img["height"], img["width"]):
        outline(ax, p, SRC, lw=2.4, ls="--", z=3)
    for i, p in enumerate(polys):
        outline(ax, p, COL[i % 3], lw=1.3, z=4, fill=COL[i % 3] + "40")
    ax.set_title(f"a  One object becomes {len(polys)}\nsupervision, COCO → YOLO", loc="left", fontsize=7)
    credit("a split", img, coco)


def panel_crowd(ax, coco, images: Path, k: int, tmp: Path):
    import supervision as sv
    c = candidates(coco, lambda a, i: a["iscrowd"] and 3000 < a["area"] < 80000)[k]
    a, img, _ = c
    j = one_image_coco(coco, img, [a], tmp / "crowd")
    sv.DetectionDataset.from_coco(str(images), str(j), force_masks=True).as_yolo(
        annotations_directory_path=str(tmp / "crowd" / "yolo"), data_yaml_path=str(tmp / "crowd" / "d.yaml"))
    polys = read_yolo_polys(next((tmp / "crowd" / "yolo").glob("*.txt")), img["width"], img["height"])
    im = Image.open(images / img["file_name"]).convert("RGB")
    crop(ax, im, a["bbox"], margin=0.1)
    for p in src_polys(a, img["height"], img["width"]):
        outline(ax, p, SRC, lw=1.6, ls="--", z=3, fill="#ffffff59")
    for i, p in enumerate(polys):
        outline(ax, p, COL[1], lw=1.2, z=4)
    n = len(polys)
    what = "an ordinary object" if n == 1 else f"{n} ordinary objects"
    ax.set_title(f"b  One crowd region becomes {what}\nsupervision, COCO → YOLO", loc="left", fontsize=7)
    credit("b crowd", img, coco)


def panel_rle(ax, coco, images: Path, k: int, tmp: Path):
    from ultralytics.data.converter import convert_coco
    c = candidates(coco, lambda a, i: not a["iscrowd"] and isinstance(a["segmentation"], list)
                   and len(a["segmentation"]) == 1 and 8000 < a["area"] < 80000
                   and a["area"] < 0.55 * a["bbox"][2] * a["bbox"][3])[k]  # irregular shape, far from its box
    a, img, _ = c
    h, w = img["height"], img["width"]
    rle = mu.merge(mu.frPyObjects(a["segmentation"], h, w))
    rle["counts"] = rle["counts"].decode()
    b = dict(a, segmentation=rle)  # the same object stored as RLE, as in our RLE-stored copy
    one_image_coco(coco, img, [b], tmp / "rle" / "ann")
    convert_coco(labels_dir=str(tmp / "rle" / "ann"), save_dir=str(tmp / "rle" / "out"), use_segments=True,
                 cls91to80=False)
    polys = read_yolo_polys(next((tmp / "rle" / "out").rglob("*.txt")), w, h)
    im = Image.open(images / img["file_name"]).convert("RGB")
    crop(ax, im, a["bbox"])
    for p in src_polys(a, h, w):
        outline(ax, p, SRC, lw=1.4, ls="--")
    for p in polys:
        outline(ax, p, COL[2], lw=1.8)
    ax.set_title("c  RLE mask becomes its box\nUltralytics, COCO → YOLO", loc="left", fontsize=7)
    credit("c rle", img, coco)


def panel_voc(ax, coco, images: Path, k: int, tmp: Path):
    import supervision as sv
    # COCO boxes are floats; supervision writes int(coordinate + 1), so the devkit reading grows the box
    c = candidates(coco, lambda a, i: not a["iscrowd"] and 8 <= a["bbox"][2] <= 20 and 8 <= a["bbox"][3] <= 20)[k]
    a, img, _ = c
    j = one_image_coco(coco, img, [a], tmp / "voc")
    sv.DetectionDataset.from_coco(str(images), str(j)).as_pascal_voc(
        annotations_directory_path=str(tmp / "voc" / "xml"))
    bb = ET.parse(next((tmp / "voc" / "xml").glob("*.xml"))).find("object/bndbox")
    xmin, ymin, xmax, ymax = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
    x, y, bw, bh = a["bbox"]
    im = Image.open(images / img["file_name"]).convert("RGB")
    crop(ax, im, a["bbox"], margin=0.6, min_size=0)
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()
    for gx in np.arange(np.floor(x0), x1 + 1):
        ax.axvline(gx, color="#d9d9d9", lw=0.25, zorder=2)
    for gy in np.arange(np.floor(y0), y1 + 1):
        ax.axhline(gy, color="#d9d9d9", lw=0.25, zorder=2)
    ax.add_patch(Rectangle((x, y), bw, bh, fill=False, ec=SRC, lw=1.2, ls="--", zorder=5))  # on top: edges coincide
    # read back with the VOC devkit convention (1-based, inclusive), as Detectron2 does
    ax.add_patch(Rectangle((xmin - 1, ymin - 1), xmax - xmin + 1, ymax - ymin + 1, fill=False, ec=COL[0], lw=1.6,
                           zorder=4))
    ax.set_title(f"d  {bw:.1f}×{bh:.1f} px box read back as {int(xmax - xmin + 1)}×{int(ymax - ymin + 1)}\n"
                 "supervision, COCO → VOC (devkit reading)", loc="left", fontsize=7)
    credit("d voc", img, coco)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="instances_val2017.json")
    ap.add_argument("--images", required=True, help="val2017 image folder")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--pick", action="append", default=[], help="panel=N, panels: split, crowd, rle, voc")
    a = ap.parse_args(argv)
    coco = json.loads(Path(a.ann).read_text())
    pick = {k: int(v) for k, v in (p.split("=") for p in a.pick)}
    fig, axes = plt.subplots(2, 2, figsize=(129 * MM, 132 * MM), gridspec_kw={"wspace": 0.06, "hspace": 0.22})
    axes = axes.ravel()
    for ax in axes:
        ax.set_anchor("N")  # top-align images in each row so the panel titles line up
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for ax, (name, fn) in zip(axes, [("split", panel_split), ("crowd", panel_crowd), ("rle", panel_rle),
                                         ("voc", panel_voc)]):
            fn(ax, coco, Path(a.images), pick.get(name, 0), tmp)
    # legend and image credits go in the figure caption
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "tiff", "png"):
        fig.savefig(out / f"fig3_examples.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("wrote", out / "fig3_examples.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
