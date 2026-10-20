"""Re-audit the Stage 2 training subset with the current measurement code (no training, no images).

Rebuilds the exact 10,000-image subset that `run_stage2.py prepare` sampled (same seed and the same
sampling call) and runs the audit on the three converted label sets used for training.

    python scripts/train_subset_audit.py --train-ann annotations/instances_train2017.json --out results_train_v2

Writes results_train_v2/summary.csv (paths: supervision_yolo and datumaro_yolo_det on the original
labels, ultralytics_yolo_seg on the RLE-stored copy, plus the faithful Ultralytics references).
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from convaudit.coco_io import load_coco, write_coco  # noqa: E402
from convaudit.stage2 import _subset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--out", default="results_train_v2")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    full = load_coco(Path(a.train_ann))
    ids = sorted(im["id"] for im in full["images"])
    chosen = set(random.Random(a.seed).sample(ids, min(a.n_train, len(ids))))
    sub = _subset(full, chosen)
    del full
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "train_subset.json"
    write_coco(sub, js)
    print(f"subset: {len(sub['images'])} images, {len(sub['annotations'])} annotations", flush=True)
    paths = "ultralytics_yolo_det,ultralytics_yolo_seg,supervision_yolo,datumaro_yolo_det"
    subprocess.run([sys.executable, str(ROOT / "run_pilot.py"), "--coco", str(js), "--out", str(out),
                    "--paths", paths, "--variants", "original,rle"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
