#!/usr/bin/env bash
# Re-run after the review fix to fragment assignment (measure.py, 2026-10-18).
# A converted piece that lies inside the boxes of two originals of the same class now goes to the
# original whose mask covers it (before: to the non-crowd original on a tie). Full conversions are
# unchanged; split and crowd counts and the split/crowd components of Stage 1 can change.
# Run from the convaudit folder (same place as run_q1_additions.sh). Takes a few hours; outputs go
# to new *_v2 folders, nothing old is overwritten.
# Usage: caffeinate -i bash scripts/rerun_after_review.sh [path/to/instances_valrem.json]
set -euo pipefail
REM="${1:-$HOME/Downloads/instances_valrem.json}"
VAL=annotations/instances_val2017.json
for f in "$VAL" "$REM" val2017 stage1/predictions; do
  [ -e "$f" ] || { echo "missing: $f (run this from the convaudit folder)"; exit 1; }
done
source .venv/bin/activate
python -m pytest -q tests

SV=supervision_yolo,supervision_voc,supervision_voc_raw
echo "== 1/4 audit: all paths on COCO val2017 (original and RLE-stored copy)"
python run_pilot.py --coco "$VAL" --images val2017 --out results_v2
echo "== 2/4 audit: all paths on COCO-ReM"
python run_pilot.py --coco "$REM" --images val2017 --out results_rem_v2 --variants original
echo "== 3/4 Stage 1: supervision paths on COCO val2017 (saved predictions reused)"
python run_stage1.py --coco "$VAL" --images val2017 --out stage1_v2 --paths "$SV" \
  --predictions-from stage1/predictions
echo "== 4/4 Stage 1: supervision paths on COCO-ReM (saved predictions reused)"
python run_stage1.py --coco "$REM" --images val2017 --out stage1_rem_v2 --paths "$SV" --variants original \
  --predictions-from stage1/predictions

tar czf review_rerun.tgz \
  results_v2/{summary.md,summary.csv,strata.csv,manifest.json} \
  results_rem_v2/{summary.md,summary.csv,strata.csv,manifest.json} \
  stage1_v2/{report.md,stage1_results.csv,manifest.json} \
  stage1_rem_v2/{report.md,stage1_results.csv,manifest.json}
python scripts/review_checks.py --coco "$VAL" --rem "$REM" --objects results_v2/objects.csv \
  --rem-objects results_rem_v2/objects.csv > review_checks_v2.txt
echo "Done. Send review_rerun.tgz and review_checks_v2.txt ($(pwd))"
