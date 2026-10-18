"""Pilot: what do common converters change in a COCO instances file?

Example:
    python run_pilot.py --coco annotations/instances_val2017.json --out results
    python run_pilot.py --coco annotations/instances_val2017.json --out quick --limit-images 200
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from convaudit import __version__
from convaudit.adapters import PATHS, run_path, tool_version
from convaudit.coco_io import (limit_images, load_coco, make_rle_variant,
                               object_table, sha256_file, shards)
from convaudit.measure import measure_image
from convaudit.report import decision, strata, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="COCO instances JSON, e.g. instances_val2017.json")
    ap.add_argument("--out", required=True, help="results folder (created)")
    ap.add_argument("--paths", default="all", help="comma list of: " + ", ".join(PATHS))
    ap.add_argument("--variants", default="original,rle", help="original and/or rle")
    ap.add_argument("--images", default=None, help="optional real image folder (else stubs are written)")
    ap.add_argument("--shard-size", type=int, default=250, help="images per conversion run")
    ap.add_argument("--limit-images", type=int, default=None, help="first N images by id, for a quick run")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(PATHS) if args.paths == "all" else [p.strip() for p in args.paths.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    images_dir = Path(args.images) if args.images else None

    base = limit_images(load_coco(args.coco), args.limit_images)
    datasets = {"original": base}
    if "rle" in variants:
        print("building RLE variant ...", flush=True)
        datasets["rle"] = make_rle_variant(base)

    manifest = {
        "convaudit": __version__,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "input": str(args.coco),
        "input_sha256": sha256_file(args.coco),
        "images": len(base["images"]),
        "annotations": len(base["annotations"]),
        "args": vars(args),
        "tools": {pkg: tool_version(pkg) for pkg in ("ultralytics", "supervision", "datumaro", "fiftyone", "pycocotools")},
        "paths": {},
    }

    all_rows, all_extras, all_added = [], [], []
    for variant in variants:
        coco = datasets[variant]
        table = object_table(coco)
        sizes = {im["file_name"]: (im["height"], im["width"], im["id"]) for im in coco["images"]}
        for path in paths:
            t0 = time.time()
            print(f"[{variant}] {path} ...", flush=True)
            meta = None
            for k, shard in enumerate(shards(coco, args.shard_size)):
                work = out / "work" / variant / path / f"shard{k:04d}"
                try:
                    result, meta = run_path(path, shard, work, images_dir)
                except Exception as e:  # record and continue with the other paths
                    manifest["paths"].setdefault(f"{variant}/{path}", {})["error"] = repr(e)
                    print(f"   failed: {e!r}", flush=True)
                    break
                for fname, (objs, bad) in result.items():
                    h, w, im_id = sizes[fname]
                    rows, extra = measure_image(table[im_id], objs, meta["kind"], h, w)
                    for r in rows:
                        r.update(variant=variant, path=path)
                    all_rows.extend(rows)
                    all_extras.append({"variant": variant, "path": path, "image_id": im_id,
                                       "n_added": extra["n_added"], "n_malformed": bad})
                    for a in extra["added"]:
                        all_added.append({"variant": variant, "path": path, "image_id": im_id, **a})
            if meta:
                manifest["paths"].setdefault(f"{variant}/{path}", {}).update(meta, seconds=round(time.time() - t0, 1))

    rows = pd.DataFrame(all_rows)
    extras = pd.DataFrame(all_extras)
    rows.to_csv(out / "objects.csv", index=False)
    pd.DataFrame(all_added, columns=["variant", "path", "image_id", "category_id", "kind",
                                     "x1", "y1", "x2", "y2"]).to_csv(out / "added.csv", index=False)
    summary = summarize(rows, extras)
    summary.to_csv(out / "summary.csv", index=False)
    st = strata(rows)
    st.to_csv(out / "strata.csv", index=False)
    verdict = decision(summary)
    by_size = st[st.stratum == "size"].pivot_table(index=["variant", "path"], columns="value",
                                                   values="material_pct").reset_index()
    (out / "summary.md").write_text(
        "# Conversion pilot summary\n\n" + summary.to_markdown(index=False) + "\n\n" + verdict
        + "\n\n## Material change (%) by object size\n\n" + by_size.to_markdown(index=False) + "\n",
        encoding="utf-8")
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print("\n" + verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
