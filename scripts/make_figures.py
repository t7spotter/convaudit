"""Regenerate every data figure of the paper from the released CSV files.

Usage:
    python scripts/make_figures.py --out figures \
        --stage2-runs report_final/stage2_runs.csv \
        [--audit coco=results/summary.csv --audit coco_rle=... --audit rem=results_rem/summary.csv] \
        [--stage1 coco=stage1/stage1_results.csv --stage1 rem=stage1_rem/stage1_results.csv]

Each figure is written as PDF (vector, for the journal), EPS and PNG (preview).
Figures whose input is not given are skipped. The qualitative examples (Fig. 3) are made by
scripts/make_examples.py, which needs the COCO images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from scipy import stats  # noqa: E402

# One palette for the whole paper (validated for colour-vision deficiency, all pairs):
# identity of the converter, never its rank. Grey is reserved for references and controls.
TOOL = {"supervision": "#2a78d6", "datumaro": "#eb6834", "ultralytics": "#1baf7a",
        "fiftyone": "#6250d6", "reference": "#6b6a66"}
MARKER = {"supervision": "o", "datumaro": "s", "ultralytics": "D", "fiftyone": "^", "reference": "o"}
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#dddcd7", "#efeeea"
MM = 1 / 25.4  # Springer widths: 84 mm (single), 129 mm, 174 mm (double)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

RULE = 0.5  # pre-specified threshold: |mean dAP| >= 0.5 and 95% CI excludes 0


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out / name)


def paired(a: pd.Series, b: pd.Series):
    d = (a - b).dropna()
    m = d.mean()
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d, m, m - h, m + h


# ---------------------------------------------------------------- Fig. 1: study design
def fig_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(174 * MM, 62 * MM))
    ax.set_xlim(0, 174)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, body="", fill="#f4f3ef", edge=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fill, ec=edge, lw=0.7))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=7.5,
                fontweight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + h - 8.2, body, ha="center", va="top", fontsize=6.5,
                    color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=INK2, shrinkA=0, shrinkB=0))

    # pipeline row
    box(2, 34, 30, 24, "Source", "COCO val2017\n+ RLE-stored copy\nCOCO-ReM val\n77,494 instances")
    box(38, 34, 34, 24, "Convert", "4 tools, 11 paths\npinned versions\nCOCO \u2192 YOLO, VOC")
    box(78, 34, 36, 24, "Read back", "independent parsers,\nnot the tools' own;\nper-object matching")
    box(120, 34, 52, 24, "RQ1  What changes?",
        "drops, splits, geometry, crowd\nregions, duplicates, box\nfallbacks (Tables 3\u20134)", fill="#e7f0fb")
    arrow(32, 46, 38, 46)
    arrow(72, 46, 78, 46)
    arrow(114, 46, 120, 46)

    # research questions that build on the read-back files
    box(60, 2, 56, 26, "RQ3  Evaluation effect",
        "fixed predictions of 5 detectors\nscored on original vs converted\nground truth (paired bootstrap)", fill="#e7f0fb")
    box(2, 2, 54, 26, "RQ4  Training effect",
        "YOLO11n(-seg) trained on each\ntool's labels, 3 seeds, scored\non the original labels", fill="#e7f0fb")
    box(120, 2, 52, 26, "RQ2  Why?", "each change traced to the\nresponsible code in the\npinned release (Table 5)",
        fill="#e7f0fb")
    arrow(96, 34, 88, 28)  # evaluation uses the read-back objects
    arrow(50, 34, 29, 28)  # training uses each tool's own files
    arrow(146, 34, 146, 28)
    save(fig, out, "fig1_study_design")


# ---------------------------------------------------------------- shared path labels
PATHS = [  # (path id, row label, tool)
    ("supervision_yolo", "supervision → YOLO", "supervision"),
    ("supervision_voc", "supervision → VOC, devkit reading", "supervision"),
    ("supervision_voc_raw", "supervision → VOC, as written", "supervision"),
    ("datumaro_yolo_det", "Datumaro → YOLO", "datumaro"),
    ("datumaro_voc", "Datumaro → VOC, devkit reading", "datumaro"),
    ("datumaro_voc_raw", "Datumaro → VOC, as written", "datumaro"),
    ("fiftyone_yolo_det", "FiftyOne → YOLO", "fiftyone"),
    ("fiftyone_voc", "FiftyOne → VOC, devkit reading", "fiftyone"),
    ("fiftyone_voc_raw", "FiftyOne → VOC, as written", "fiftyone"),
    ("ultralytics_yolo_det", "Ultralytics → YOLO detection", "ultralytics"),
    ("ultralytics_yolo_seg", "Ultralytics → YOLO segmentation", "ultralytics"),
]
# a fourth tool hue cannot pass the all-pairs colour checks next to the other three, so FiftyOne is ink
PATH_COLOR = {"supervision": TOOL["supervision"], "datumaro": TOOL["datumaro"],
              "ultralytics": TOOL["ultralytics"], "fiftyone": "#3d3c39"}


def _label_rows(ax, labels):
    rows = labels
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.4, -0.6)  # first row on top
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------- Fig. 2: how often objects change
def fig_prevalence(audit: dict[str, pd.DataFrame], out: Path) -> None:
    """audit: dataset name -> summary.csv frame. Names: coco, coco_fiftyone, rem."""
    sets = [  # (label, frame key, variant, marker, fill, vertical offset)
        ("COCO val2017 (polygons)", ["coco", "coco_fiftyone"], "original", "o", True, -0.26),
        ("COCO val2017, RLE-stored copy", ["coco", "coco_fiftyone"], "rle", "s", False, 0.0),
        ("COCO-ReM val (all RLE)", ["rem"], "original", "^", True, 0.26),
    ]
    fig, ax = plt.subplots(figsize=(150 * MM, 84 * MM))
    for y, (pid, _, tool) in enumerate(PATHS):
        vals = []
        for label, keys, variant, mk, filled, dy in sets:
            for k in keys:
                f = audit.get(k)
                if f is None:
                    continue
                r = f[(f.variant == variant) & (f.path == pid)]
                if len(r):
                    v = float(r["material_%"].iloc[0])
                    vals.append(v)
                    c = PATH_COLOR[tool]
                    ax.scatter([v], [y + dy], s=22, marker=mk, facecolor=c if filled else "white", edgecolor=c,
                               lw=1.0, zorder=3)
        if vals:
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    _label_rows(ax, [r[1] for r in PATHS])
    ax.set_xlim(-2, 100)
    ax.set_xlabel("Objects materially changed (%)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for label, _, _, mk, filled, _ in sets:
        ax.scatter([], [], s=22, marker=mk, facecolor=INK2 if filled else "white", edgecolor=INK2, lw=1.0,
                   label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=6.5,
              handletextpad=0.3, columnspacing=1.4)
    save(fig, out, "fig2_prevalence")


# ---------------------------------------------------------------- Fig. 4: evaluation effects
COMPONENTS = [("split", "split objects", "#6250d6"), ("crowd", "crowd regions", "#eda100"),
              ("geometry", "geometry", "#e87ba4")]  # validated trio, distinct from the tool hues


def fig_evaluation_effects(stage1: pd.DataFrame, out: Path) -> None:
    """stage1: concatenated stage1_results.csv files (all paths, variants original and rle)."""
    def rows_for(iou, spec):
        out_rows = []
        for variant, pid, label in spec:
            s = stage1[(stage1.variant == variant) & (stage1.path == pid) & (stage1.iou_type == iou)]
            if not len(s):
                continue
            full = s[s.component == "full"].dAP
            comps = {c: s[s.component == c].dAP.mean() if (s.component == c).any() else 0.0
                     for c, _, _ in COMPONENTS}
            out_rows.append((label, full.mean(), full.min(), full.max(), comps, full.size))
        return out_rows

    box_spec = [("original", p, lab) for p, lab, _ in PATHS]
    mask_spec = [("original", "supervision_yolo", "supervision → YOLO"),
                 ("rle", "supervision_yolo", "supervision → YOLO, RLE-stored copy"),
                 ("original", "ultralytics_yolo_seg", "Ultralytics → YOLO segmentation"),
                 ("rle", "ultralytics_yolo_seg", "Ultralytics → YOLO seg., RLE-stored copy")]
    panels = [("a  Box AP, 5 detectors", rows_for("bbox", box_spec), (-15, 0.8)),
              ("b  Mask AP, 2 segmentation models", rows_for("segm", mask_spec), (-30, 1.6))]
    fig, axes = plt.subplots(2, 1, figsize=(150 * MM, 112 * MM),
                             gridspec_kw={"height_ratios": [len(panels[0][1]), len(panels[1][1])], "hspace": 0.45})
    for ax, (title, rows, xlim) in zip(axes, panels):
        for y, (label, m, lo, hi, comps, n) in enumerate(rows):
            left = 0.0
            for c, _, col in COMPONENTS:  # single-change rebuilds, stacked leftwards from zero
                v = min(comps[c], 0.0)
                if v < -0.02:
                    ax.barh(y, v, left=left, height=0.5, color=col, edgecolor="white", lw=1.0, zorder=2)
                    left += v
            ax.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=3)
            ax.scatter([m], [y], s=22, color=INK, zorder=4, edgecolor="white", lw=0.7)
            ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), y, f"{m:+.1f}".replace("-", "\u2212"), va="center", ha="left", fontsize=6.3,
                    color=INK2)
        _label_rows(ax, [r[0] for r in rows])
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
    handles = [Patch(color=col, label=f"only {lab} changed") for _, lab, col in COMPONENTS]
    handles.append(Line2D([0], [0], color=INK, lw=1.0, marker="o", markersize=4.5, markeredgecolor="white",
                          label="full conversion (mean; line = range across models)"))
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=2, frameon=False,
                   fontsize=6.3, handlelength=1.4, columnspacing=1.2)
    axes[-1].set_xlabel("dAP: AP against the converted ground truth minus AP against the original (points)")
    save(fig, out, "fig4_evaluation_effects")


# ---------------------------------------------------------------- Fig. 5: training effects
COMPARISONS = [  # (label, condition, reference, iou_type, tool)
    ("Datumaro (boxes)", "det_datumaro", "det_reference", "bbox", "datumaro"),
    ("supervision (polygons)", "det_supervision", "det_reference_poly", "bbox", "supervision"),
    ("Control: polygons vs boxes", "det_reference_poly", "det_reference", "bbox", "reference"),
    ("supervision (polygons)", "seg_supervision", "seg_reference", "bbox", "supervision"),
    ("Ultralytics on RLE masks", "seg_ultralytics_rle", "seg_reference", "bbox", "ultralytics"),
    ("supervision (polygons)", "seg_supervision", "seg_reference", "segm", "supervision"),
    ("Ultralytics on RLE masks", "seg_ultralytics_rle", "seg_reference", "segm", "ultralytics"),
]


def fig_training_effects(runs: pd.DataFrame, out: Path) -> None:
    o = runs[runs.eval_gt == "original"]

    def ap(cond, iou):
        return o[(o.condition == cond) & (o.iou_type == iou)].set_index("seed").AP

    panels = [("a  Detection models, box AP", COMPARISONS[0:3], (-2.2, 1.2)),
              ("b  Segmentation models, box AP", COMPARISONS[3:5], (-2.2, 1.2)),
              ("c  Segmentation models, mask AP", COMPARISONS[5:7], (-9.5, 1.2))]
    fig, axes = plt.subplots(3, 1, figsize=(150 * MM, 80 * MM),
                             gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.95})
    for ax, (title, comps, xlim) in zip(axes, panels):
        ax.axvspan(-RULE, RULE, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for i, (label, cond, ref, iou, tool) in enumerate(comps):
            y = len(comps) - 1 - i
            d, m, lo, hi = paired(ap(cond, iou), ap(ref, iou))
            meaningful = abs(m) >= RULE and (lo > 0 or hi < 0)
            c = TOOL[tool]
            ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter(d.values, [y + 0.28] * len(d), s=9, facecolor="white", edgecolor=c,
                       lw=0.7, marker=MARKER[tool], zorder=3)
            ax.scatter([m], [y], s=34, color=c, marker=MARKER[tool], zorder=4,
                       edgecolor="white", lw=0.8)
            txt = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if meaningful else "")
            txt = txt.replace("-", "\u2212")
            ax.annotate(txt, xy=(1.02, y), xycoords=("axes fraction", "data"), ha="left", va="center",
                        fontsize=6.5, color=INK if meaningful else INK2,
                        fontweight="bold" if meaningful else "normal", annotation_clip=False)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([c[0] for c in comps][::-1])
        ax.set_ylim(-0.6, len(comps) - 0.4)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
    axes[0].annotate("mean [95% CI]", xy=(1.02, 1.08), xycoords="axes fraction", ha="left",
                     fontsize=6.5, color=INK2)
    axes[-1].set_xlabel("dAP = condition minus its reference (AP points)")
    save(fig, out, "fig5_training_effects")


# ---------------------------------------------------------------- Fig. 6: matched conversion
def fig_matched_conversion(runs: pd.DataFrame, out: Path) -> None:
    s = runs[(runs.iou_type == "segm") & runs.condition.isin(["seg_reference", "seg_ultralytics_rle"])]
    gts = [("original", "Original\nmasks"), ("converted:ultralytics_yolo_det", "Masks reduced\nto boxes")]
    lines = [("seg_reference", "Trained on faithful masks", "reference"),
             ("seg_ultralytics_rle", "Trained on RLE masks\nconverted to boxes", "ultralytics")]
    fig, ax = plt.subplots(figsize=(84 * MM, 62 * MM))
    xs = np.arange(len(gts))
    for k, (cond, label, tool) in enumerate(lines):
        means = []
        for x, (g, _) in zip(xs, gts):
            v = s[(s.condition == cond) & (s.eval_gt == g)].AP.values
            means.append(v.mean())
            # seed marks set a little to the side of the line so the mean marker does not hide them
            ax.scatter(np.full(len(v), x + (0.07 if k else -0.07)), v, s=10, facecolor="white",
                       edgecolor=TOOL[tool], lw=0.7, marker=MARKER[tool], zorder=3)
        ax.plot(xs, means, color=TOOL[tool], lw=1.8, marker=MARKER[tool], ms=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.text(xs[-1] + 0.08, means[-1], f"{label}\n{means[-1]:.1f}", va="center", fontsize=6.5,
                color=INK)
        ax.text(xs[0] - 0.08, means[0], f"{means[0]:.1f}", va="center", ha="right", fontsize=6.5,
                color=INK)
    # paired difference under the matched conversion (same seeds)
    g = gts[-1][0]
    a = s[(s.condition == "seg_ultralytics_rle") & (s.eval_gt == g)].set_index("seed").AP
    b = s[(s.condition == "seg_reference") & (s.eval_gt == g)].set_index("seed").AP
    _, m, lo, hi = paired(a, b)
    ax.annotate("", xy=(xs[-1] - 0.16, a.mean()), xytext=(xs[-1] - 0.16, b.mean()),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK2, mutation_scale=7))
    ax.text(xs[-1] - 0.12, (a.mean() + b.mean()) / 2, f"{m:+.1f}\n[{lo:.1f}, {hi:.1f}]".replace("-", "\u2212"),
            ha="left", va="center", fontsize=6.3, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[1] for g in gts])
    ax.set_xlim(-0.45, 2.2)
    ax.set_ylim(0, 16)
    ax.set_ylabel("Mask AP (mean of 3 seeds; open marks = seeds)")
    ax.set_xlabel("Evaluation ground truth")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, out, "fig6_matched_conversion")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--stage2-runs")
    ap.add_argument("--audit", action="append", default=[], help="name=path/to/summary.csv")
    ap.add_argument("--stage1", action="append", default=[], help="name=path/to/stage1_results.csv")
    a = ap.parse_args(argv)
    out = Path(a.out)
    fig_overview(out)
    if a.audit:
        fig_prevalence({k: pd.read_csv(v) for k, v in (s.split("=", 1) for s in a.audit)}, out)
    if a.stage1:
        fig_evaluation_effects(pd.concat([pd.read_csv(s.split("=", 1)[1]) for s in a.stage1]), out)
    if a.stage2_runs:
        runs = pd.read_csv(a.stage2_runs)
        fig_training_effects(runs, out)
        fig_matched_conversion(runs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
