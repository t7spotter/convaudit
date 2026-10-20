#!/usr/bin/env bash
# Q1 additions on the Mac: FiftyOne (4th tool) and COCO-ReM (real RLE masks).
# Run from the convaudit folder that already holds annotations/, val2017/ and stage1/.
# Usage: bash scripts/run_q1_additions.sh [path/to/instances_valrem.json]
set -euo pipefail
REM="${1:-$HOME/Downloads/instances_valrem.json}"
VAL=annotations/instances_val2017.json
for f in "$VAL" "$REM" val2017 stage1/predictions; do
  [ -e "$f" ] || { echo "missing: $f (run this from the convaudit folder)"; exit 1; }
done
source .venv/bin/activate
uv pip install -q fiftyone==1.22.1
python -m pytest -q tests

FO=fiftyone_yolo_det,fiftyone_voc,fiftyone_voc_raw
echo "== 1/4 audit: FiftyOne on COCO val2017"
python run_pilot.py --coco "$VAL" --images val2017 --out results_fiftyone --paths "$FO"
echo "== 2/4 audit: all tools on COCO-ReM"
python run_pilot.py --coco "$REM" --images val2017 --out results_rem --variants original
echo "== 3/4 Stage 1: FiftyOne on COCO val2017 (saved predictions reused)"
python run_stage1.py --coco "$VAL" --images val2017 --out stage1_fiftyone --paths "$FO" \
  --predictions-from stage1/predictions
echo "== 4/4 Stage 1: all tools on COCO-ReM (saved predictions reused)"
python run_stage1.py --coco "$REM" --images val2017 --out stage1_rem --variants original \
  --predictions-from stage1/predictions

tar czf q1_results.tgz \
  results_fiftyone/{summary.md,summary.csv,strata.csv,added.csv,manifest.json} \
  results_rem/{summary.md,summary.csv,strata.csv,added.csv,manifest.json} \
  stage1_fiftyone/{report.md,stage1_results.csv,manifest.json} \
  stage1_rem/{report.md,stage1_results.csv,manifest.json}
echo "Done. Send q1_results.tgz ($(pwd)/q1_results.tgz)"
