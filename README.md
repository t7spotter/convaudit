# convaudit: what do annotation converters change?

Replication package for the paper *Lost in Conversion: Silent Label Changes in Annotation Format
Converters and Their Effect on Object Detection and Instance Segmentation Results* (Mohamadali
Firouzi, 2026; submitted).

The harness converts a COCO instance-annotation file with each tool (Ultralytics, Roboflow
supervision, Datumaro, FiftyOne), reads the output back with its **own** parsers (never the
tools' readers), matches every written object to its original, and counts what changed (audit,
`run_pilot.py`). It then measures what the changes do to AP with fixed predictions
(`run_stage1.py`) and to models trained on the converted labels (`run_stage2.py`).
`scripts/make_figures.py` and `scripts/make_examples.py` rebuild the paper's figures, and
`repros/` holds the minimal reproductions sent to the tools' maintainers.

## Quick start (macOS, Apple Silicon)

```bash
# 1. Python 3.12 environment (uv downloads Python itself)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd convaudit
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Check the harness (about 10 s)
python -m pytest -q tests

# 3. COCO 2017 annotations only; no images needed (about 250 MB download)
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip annotations/instances_val2017.json

# 4. Quick run on 200 images (about 1 minute), then the full run (about 15 minutes)
python run_pilot.py --coco annotations/instances_val2017.json --out results_quick --limit-images 200
python run_pilot.py --coco annotations/instances_val2017.json --out results
```

Send back `results/summary.md`, `results/summary.csv`, `results/strata.csv` and
`results/manifest.json`. `results/objects.csv` (one row per object) is useful too but large.

## Conversion paths

| Path | Tool and call | Output | Read back as |
|---|---|---|---|
| `ultralytics_yolo_det` | `ultralytics.data.converter.convert_coco(use_segments=False)` | YOLO boxes | boxes |
| `ultralytics_yolo_seg` | `convert_coco(use_segments=True)` | YOLO polygons | masks |
| `supervision_yolo` | `sv.DetectionDataset.from_coco(...).as_yolo(...)` | YOLO polygons | masks |
| `supervision_voc` / `_raw` | `...as_pascal_voc(...)` | VOC XML | boxes |
| `datumaro_yolo_det` | `dm.Dataset.import_from(..., "coco_instances").export(..., "yolo")` | YOLO boxes | boxes |
| `datumaro_voc` / `_raw` | `...export(..., "voc_detection")` | VOC XML | boxes |
| `fiftyone_yolo_det` | `fo.Dataset.from_dir(..., COCODetectionDataset).export(..., YOLOv5Dataset, classes=...)` | YOLO boxes | boxes |
| `fiftyone_voc` / `_raw` | `...export(..., VOCDetectionDataset)` | VOC XML | boxes |

FiftyOne's YOLO export writes boxes only (instance masks are not turned into polygons), so it
has no segmentation path. The class list is passed explicitly in category-id order, as FiftyOne's
documentation advises.

Each path runs on two inputs:

- **original**: the COCO file as published.
- **rle**: the same file with every non-crowd polygon stored as an RLE mask of exactly the same
  pixels. This mimics datasets exported by tools that save masks as RLE (brush or SAM masks, for
  example). Any difference on this variant comes from how a converter handles RLE.

**VOC is read twice.** The VOC devkit defines boxes as 1-based, inclusive pixel indices, and
standard loaders read them that way (Detectron2's loader subtracts 1 from `xmin` and `ymin`).
Tools disagree on what they write, so each VOC output is measured under the devkit reading
(`*_voc`) and as written (`*_voc_raw`).

**Stub images.** supervision opens every image to get its size. Without `--images`, the harness
writes black JPEGs of the right sizes. Annotation geometry does not depend on pixel values. Pass
`--images` with the real `val2017` folder to remove even that assumption.

## How changes are measured

All rasterization uses pycocotools, the reference used by COCO evaluation, for originals and
converted objects alike.

Per image:

1. Converted boxes (or polygon extents) are clipped to the image, as training loaders do.
2. **Matching**:
   - Non-crowd originals are matched to converted objects by box IoU (Hungarian, IoU ≥ 0.5).
   - A second pass matches leftovers of the same class at IoU ≥ 0.1, so a badly moved object
     counts as *changed*, not *dropped plus added*.
   - Crowd regions are matched next.
3. **Fragments**: a leftover converted object of the same class lying ≥ 90% inside an original
   (2 px tolerance) is a fragment, so one object was written as several. An original with no
   match but with fragments counts as *split*, not *dropped*.
4. Anything still unmatched is **added**. An original with nothing is **dropped**.

Per original object, `objects.csv` records:

- status: `matched`, `dropped`, `crowd_kept` or `crowd_dropped`;
- class change;
- box IoU and largest corner deviation in px;
- for polygon outputs: mask IoU of the object as written, and mask IoU of all its fragments
  together;
- box fallback: a mask replaced by its bounding rectangle;
- number of fragments.

It also records descriptors used for stratification: crowd, polygon/RLE/none, number of parts,
holes, COCO size class (small < 32², medium < 96², large), and touching the border.

**Material change** for an object means any of these:

- it was dropped;
- its class changed;
- its geometry IoU fell below 0.9 (mask IoU for polygon outputs, box IoU otherwise);
- it was split into several objects;
- for a crowd region, it was written as an ordinary object with no crowd or `difficult` flag.

## Pre-specified decision rule

Written on 2026-10-14, before any run on real COCO data:

> **GO** to a downstream training study if at least one conversion path materially changes
> ≥ 5% of all objects in the **original** COCO val2017 instances file.
> **CONDITIONAL** if only the RLE variant crosses 5%. The study would then rest on datasets that
> store masks as RLE, which needs its own justification.
> **STOP or reframe** if no path crosses 5% on either input.

GO only means the downstream test is worth running. That test trains the same model on original
and converted labels with several seeds, and the effect must exceed seed-to-seed variation.

Read the size breakdown before the headline number. One pixel moves the IoU of a 6 px object a lot.
`material_%_no_small` gives the same rate without small objects.

## Outputs

| File | Content |
|---|---|
| `summary.md` / `summary.csv` | One row per input × path, the decision, and a by-size table |
| `strata.csv` | Material-change rate by crowd, mask type, multi-part, holes, size, border |
| `objects.csv` | One row per original object per path |
| `manifest.json` | Input SHA-256, tool versions, Python, platform, arguments, timings, any errors |
| `work/` | Raw tool output, kept for inspection |

## Stage 1: effect on measured AP (`run_stage1.py`)

**Question.** If a model's predictions stay fixed and only the evaluation ground truth has passed
through a converter, how much does COCO AP change, and which defect type causes it?

**Design.**

1. Five pretrained models are run once on COCO val2017 and their predictions saved:
   - torchvision Faster R-CNN v2 (boxes);
   - torchvision Mask R-CNN v2 (boxes and masks);
   - Ultralytics YOLO11s (boxes);
   - YOLO11s-seg (boxes and masks);
   - RT-DETR-L (boxes).
2. Each conversion path's output is rebuilt as a COCO file. This is done once for the whole
   conversion (`full`) and once for each defect type alone, with everything else original:
   - `crowd`: crowd regions dropped or written as ordinary objects;
   - `split`: one object written as several;
   - `geometry`: changed coordinates or outlines, original area kept;
   - `area`: area recomputed from what was written; box formats cannot store mask area;
   - `dropped`: objects lost.
3. Every model is scored with pycocotools against the original and every rebuilt file: box AP
   for all paths, mask AP for paths that write polygons. Files that are identical for evaluation
   purposes are scored once.
4. `dAP = AP(converted) − AP(original)`, in AP points (0–100), with the same predictions. The
   uncertainty is a paired bootstrap over images: 1,000 resamples, the same resamples for both.
   This harness's AP is checked against pycocotools' official number on every run, and the
   report states the largest difference (it should be 0).

**Pre-specified rule** (written 2026-10-14, before any run on real data): a conversion effect is
**meaningful** if |dAP| ≥ 0.5 AP points and its 95% bootstrap interval excludes 0. The interval
is computed only where |dAP| ≥ 0.5, since smaller effects fail the rule regardless.

**Run** (needs the val2017 images, about 780 MB):

```bash
curl -LO http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
# quick check, about 15 minutes including model downloads
python run_stage1.py --coco annotations/instances_val2017.json --images val2017 --out stage1_quick --limit-images 200 --bootstrap 100
# full run, a few hours; predictions resume if interrupted
python run_stage1.py --coco annotations/instances_val2017.json --images val2017 --out stage1
```

**Reusing predictions.** Predictions depend only on the images, so a later run on the same images
with other paths or another annotation file can reuse them:

```bash
# FiftyOne paths on COCO val2017
python run_stage1.py --coco annotations/instances_val2017.json --images val2017 --out stage1_fiftyone \
  --paths fiftyone_yolo_det,fiftyone_voc,fiftyone_voc_raw --predictions-from stage1/predictions
# COCO-ReM (refined val2017 masks, all stored as RLE): every path, original input only
python run_stage1.py --coco instances_valrem.json --images val2017 --out stage1_rem \
  --variants original --predictions-from stage1/predictions
```

**Outputs** in the run folder:

| File | Content |
|---|---|
| `report.md` | dAP of each full conversion per model, then every component with intervals |
| `stage1_results.csv` | Every model × IoU type × ground-truth file: AP, AP50, AP75, APs, APm, APl, before and after |
| `gt/` | All rebuilt ground-truth files, with `index.csv` |
| `predictions/` | Saved model predictions |
| `manifest.json` | Tool versions, input hash, timings |

## Stage 2: effect of converted *training* labels (`run_stage2.py`)

**Question.** If a model is trained on labels that passed through a converter, how much does
its accuracy change when it is evaluated against the original COCO val2017 labels?

**Design.** Only the training labels differ between conditions:

1. **Data:** one fixed random subset of 10,000 COCO train2017 images (seed 0). The same images are
   used in every condition.
2. **Label sets:** each tool converts that subset, and its own YOLO label files are used unchanged.
3. **Conditions** (each compared with the reference in the last column):

| Condition | Task | Training labels | Compared with |
|---|---|---|---|
| `det_reference` | detection | Ultralytics `convert_coco` boxes (audited: faithful; drops crowd regions, as COCO training does) | — |
| `det_reference_poly` | detection | Ultralytics `convert_coco(use_segments=True)` polygons, same faithful labels | — (control vs `det_reference`) |
| `det_supervision` | detection | supervision `as_yolo` polygons (splits multi-part objects, writes crowd regions as objects) | `det_reference_poly` |
| `det_datumaro` | detection | Datumaro `yolo` boxes (writes crowd regions as objects) | `det_reference` |
| `seg_reference` | segmentation | Ultralytics `convert_coco(use_segments=True)` | — |
| `seg_supervision` | segmentation | supervision `as_yolo` | `seg_reference` |
| `seg_ultralytics_rle` | segmentation | Ultralytics on the RLE-stored copy (masks become rectangles) | `seg_reference` |

4. **Model and recipe:** YOLO11n or YOLO11n-seg, with the backbone initialized from ImageNet-pretrained
   YOLO11n-cls. Identical recipe for every run: 30 epochs, 640 px, batch 16, SGD lr 0.01,
   deterministic mode. Seeds 0, 1 and 2, so 21 runs (about 40 T4 hours).
5. **Evaluation:** predictions on all 5,000 val2017 images (conf 0.001, max 100 per image), scored
   with pycocotools against the **original** ground truth. Class indices are mapped through each
   tool's own class names. Optionally, runs are also scored against Stage 1's converted ground
   truth, giving a train-label × evaluation-label matrix.

**Protocol amendment, 2026-10-15** (made after `det_reference` seed 0 finished and while
`det_supervision` seed 0 was training, before any comparison was computed): supervision writes
polygon lines, and when a YOLO label file holds polygons, Ultralytics derives each training box
from the polygon after augmentation instead of transforming a stored box. A detector trained on
supervision's labels therefore differs from the box-label reference in two ways: the conversion
defects and the label format. Epoch time showed the difference (4:16 vs 2:17 per epoch on a T4).
To isolate the conversion effect, `det_reference_poly` was added: the same faithful Ultralytics
labels written as polygons. supervision is compared with it, Datumaro (boxes) with
`det_reference`. `det_reference_poly` vs `det_reference` is reported as a control: the effect of
label format alone. No run was discarded; the original 18 runs are unchanged and 3 were added.

Each run's `done.json` records the GPU it finished on and every resume (`resumes.json`: the epoch it
resumed after, the GPU and the time), so hardware consistency can be checked afterwards.

**Pre-specified rule** (written 2026-10-15, before any training run): a condition differs
meaningfully from its reference if the mean seed-paired AP difference is at least 0.5 AP points in
magnitude and its 95% t-interval (paired by seed) excludes 0.

**Hardware consistency.** Seed-major order (all conditions for seed 0, then seed 1, then seed 2)
means a partial budget still gives complete comparisons. If two machines are used, give each whole
seeds (for example `--seeds 2` on the second machine), so paired runs share the same GPU.

**Run on Colab:** `notebooks/stage2_colab.ipynb` (instructions inside). From a shell:

```bash
python run_stage2.py prepare --train-ann annotations/instances_train2017.json --val-ann annotations/instances_val2017.json --data stage2_data
python run_stage2.py train --data stage2_data --project stage2_runs --val-images val2017
python run_stage2.py status --project stage2_runs
python run_stage2.py analyze --project stage2_runs --data stage2_data --out stage2_report --stage1-gt stage1/gt
```

`prepare` also writes `audit/train_subset_audit.csv`: what each tool changed in the training subset.

## Paper figures

All data figures are rebuilt from the released CSV files (PDF, EPS and PNG, Springer widths):

```bash
python scripts/make_figures.py --out figures \
    --stage2-runs stage2_report/stage2_runs.csv \
    --audit coco=results/summary.csv --audit coco_fiftyone=results_fiftyone/summary.csv \
    --audit rem=results_rem/summary.csv \
    --stage1 coco=stage1/stage1_results.csv --stage1 fiftyone=stage1_fiftyone/stage1_results.csv
```

Fig. 3 (real examples) needs the COCO val2017 images and uses only CC BY images; it prints a credit
line for each image it uses:

```bash
python scripts/make_examples.py --ann annotations/instances_val2017.json --images val2017 --out figures
```

## Known limitations

- Only COCO-format input and four tools so far. Adding a tool is one function in
  `convaudit/adapters.py`.
- Keypoints, oriented boxes (DOTA) and round trips back to COCO are not covered yet.
- The 0.5 / 0.1 / 0.9 thresholds are choices. Per-object IoUs are in `objects.csv`, so other
  thresholds can be applied afterwards without re-running.
- `tests/make_synthetic_coco.py` makes random geometry for timing checks only. Never use it for
  results.

## Tests

`tests/test_harness.py` checks four things:

- an exact copy reports no change;
- injected defects are each detected: 12 px shift, wrong class, dropped object, crowd kept,
  unrelated extra object, split object, mask replaced by its box, tiny-object shift, and
  out-of-image box;
- the RLE variant keeps every pixel;
- the parsers follow the format conventions, and an end-to-end run on a 7-object fixture works.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The converters it
tests (Ultralytics, supervision, Datumaro, FiftyOne) are not included; they are installed separately
from `requirements.txt` under their own licences, which apply to them (Ultralytics is AGPL-3.0).
Derived data (audit tables, rebuilt ground truth, predictions, run records) is released under CC BY 4.0,
following the licences of COCO 2017 annotations and COCO-ReM.
