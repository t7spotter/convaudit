"""Stage 2 command line: prepare, train (queue), status, analyze.

Examples (see README and notebooks/stage2_colab.ipynb):
    python run_stage2.py prepare --train-ann annotations/instances_train2017.json \
        --val-ann annotations/instances_val2017.json --data stage2_data
    python run_stage2.py train --data stage2_data --project stage2_runs --val-images val2017
    python run_stage2.py status --project stage2_runs
    python run_stage2.py analyze --project stage2_runs --data stage2_data --out stage2_report \
        [--stage1-gt stage1/gt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convaudit import stage2
from convaudit.stage2_analysis import analyze


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--train-ann", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-images", default=None, help="optional local train2017 folder (else downloaded)")
    p.add_argument("--val-images", default=None, help="optional local val2017 folder (else downloaded)")
    p.add_argument("--labelsets", default=None, help="comma list (default: all)")

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--project", required=True)
    t.add_argument("--val-images", required=True, help="val2017 image folder for final predictions")
    t.add_argument("--conditions", default=None)
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--device", default=None)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--no-pretrained", action="store_true", help="tests only")
    t.add_argument("--epochs", type=int, default=None, help="tests only")
    t.add_argument("--imgsz", type=int, default=None, help="tests only")
    t.add_argument("--batch", type=int, default=None, help="tests only")

    s = sub.add_parser("status")
    s.add_argument("--project", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("--project", required=True)
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--stage1-gt", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        stage2.prepare(Path(args.train_ann), Path(args.val_ann), Path(args.data), args.n_train, args.seed,
                       Path(args.train_images) if args.train_images else None,
                       Path(args.val_images) if args.val_images else None,
                       args.labelsets.split(",") if args.labelsets else None)
    elif args.cmd == "train":
        over = {k: v for k, v in (("epochs", args.epochs), ("imgsz", args.imgsz), ("batch", args.batch)) if v}
        stage2.queue(Path(args.data), Path(args.project), Path(args.val_images),
                     args.conditions.split(",") if args.conditions else None,
                     tuple(int(x) for x in args.seeds.split(",")),
                     device=args.device, workers=args.workers, pretrained=not args.no_pretrained,
                     hyp_overrides=over)
    elif args.cmd == "status":
        print(stage2.status(Path(args.project)).to_string(index=False))
    elif args.cmd == "analyze":
        analyze(Path(args.project), Path(args.data) / "coco" / "val2017.json", Path(args.out),
                Path(args.stage1_gt) if args.stage1_gt else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
