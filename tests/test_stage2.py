"""Stage 2 end to end on a tiny synthetic dataset, CPU, one epoch (about a minute)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import make_synthetic_coco  # noqa: E402


def _dataset(tmp: Path, name: str, n: int, seed: int) -> tuple[Path, Path]:
    f = tmp / f"{name}.json"
    sys.argv = ["x", str(f), "--images", str(n), "--seed", str(seed)]
    make_synthetic_coco.main()
    c = json.loads(f.read_text())
    imgs = tmp / f"{name}_imgs"
    imgs.mkdir()
    by = {}
    for a in c["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    for im in c["images"]:
        img = np.full((im["height"], im["width"], 3), 40, np.uint8)
        for a in by.get(im["id"], []):
            s = a["segmentation"]
            r = mu.merge(mu.frPyObjects(s, im["height"], im["width"])) if isinstance(s, list) \
                else mu.frPyObjects(s, *s["size"])
            img[mu.decode(r) > 0] = (37 * a["category_id"]) % 255
        cv2.imwrite(str(imgs / im["file_name"]), img)
        im["coco_url"] = "http://invalid.local/" + im["file_name"]
    f.write_text(json.dumps(c))
    return f, imgs


def test_stage2_pipeline(tmp_path):
    tr, tr_imgs = _dataset(tmp_path, "train", 12, 5)
    va, va_imgs = _dataset(tmp_path, "val", 8, 6)
    data, runs, rep = tmp_path / "data", tmp_path / "runs", tmp_path / "rep"
    py = [sys.executable, "-W", "ignore", str(ROOT / "run_stage2.py")]
    subprocess.run(py + ["prepare", "--train-ann", str(tr), "--val-ann", str(va), "--data", str(data),
                         "--n-train", "10", "--train-images", str(tr_imgs), "--val-images", str(va_imgs),
                         "--labelsets", "reference_det,reference_seg,supervision"], check=True, cwd=ROOT)
    assert (data / "labelsets" / "supervision" / "train").exists()
    for seed in ("0", "1"):
        subprocess.run(py + ["train", "--data", str(data), "--project", str(runs), "--val-images", str(va_imgs),
                             "--seeds", seed, "--conditions", "det_reference,det_reference_poly,det_supervision", "--device", "cpu",
                             "--workers", "0", "--no-pretrained", "--epochs", "1", "--imgsz", "64", "--batch", "4"],
                       check=True, cwd=ROOT, capture_output=True)
    subprocess.run(py + ["analyze", "--project", str(runs), "--data", str(data), "--out", str(rep)],
                   check=True, cwd=ROOT)
    cmp = pd.read_csv(rep / "stage2_comparison.csv")
    row = cmp[cmp.condition == "det_supervision"].iloc[0]
    assert row["reference"] == "det_reference_poly" and row["role"] == "conversion"
    assert row["paired_seeds"] == 2 and "dAP_ci_low" in cmp
    ctrl = cmp[(cmp.condition == "det_reference_poly") & (cmp.role == "control")].iloc[0]
    assert ctrl["reference"] == "det_reference" and ctrl["paired_seeds"] == 2
    assert (runs / "det_supervision_s0" / "done.json").exists()
    done = json.loads((runs / "det_supervision_s0" / "done.json").read_text())
    assert "gpu_at_finish" in done and "resumes" in done
