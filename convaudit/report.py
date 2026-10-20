"""Summaries and the pre-specified go/stop check."""

from __future__ import annotations

import numpy as np
import pandas as pd

GO_THRESHOLD = 0.05  # pre-specified: share of all original objects materially changed


def summarize(rows: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (variant, path), g in rows.groupby(["variant", "path"], sort=False):
        e = extras[(extras.variant == variant) & (extras.path == path)]
        nc = g[g.iscrowd == 0]
        cr = g[g.iscrowd == 1]
        m = nc[nc.status == "matched"]
        geom = m["mask_iou"] if "mask_iou" in m and m["mask_iou"].notna().any() else m["box_iou"]
        n = len(g)
        rec = {
            "variant": variant,
            "path": path,
            "objects": n,
            "non_crowd": len(nc),
            "crowd": len(cr),
            "dropped_%": _pct((nc.status == "dropped").sum(), len(nc)),
            "class_changed_%": _pct((m.class_ok == False).sum(), len(nc)),  # noqa: E712
            "geometry_changed_%": _pct(m.get("geom_changed", pd.Series(dtype=object)).eq(True).sum(), len(nc)),
            "split_%": _pct((m.n_fragments > 0).sum(), len(nc)),
            "box_fallback_%": _pct(m.get("rect_fallback", pd.Series(dtype=object)).eq(True).sum(), len(nc)),
            "crowd_kept_%": _pct((cr.status == "crowd_kept").sum(), len(cr)),
            "added_per_100": round(100 * e.n_added.sum() / n, 2) if n else 0.0,
            "malformed_lines": int(e.n_malformed.sum()),
            "median_iou": round(float(np.nanmedian(geom)), 4) if len(geom) else np.nan,
            "p5_iou": round(float(np.nanpercentile(geom, 5)), 4) if len(geom) else np.nan,
            "material_%": _pct(g.material.sum(), n),
            # same rate without small objects (< 32x32 px), where 1 px moves IoU a lot
            "material_%_no_small": _pct(g[g["size"] != "small"].material.sum(), (g["size"] != "small").sum()),
        }
        rec["go"] = rec["material_%"] >= 100 * GO_THRESHOLD
        out.append(rec)
    return pd.DataFrame(out)


def strata(rows: pd.DataFrame) -> pd.DataFrame:
    """Material-change rate by object type, to see where changes concentrate."""
    r = rows.copy()
    r["multi_part"] = r.n_parts > 1
    parts = []
    for col in ["iscrowd", "seg_type", "multi_part", "has_holes", "size", "touches_border"]:
        t = (r.groupby(["variant", "path", col]).material
             .agg(objects="size", material_pct=lambda s: round(100 * s.mean(), 2))
             .reset_index().rename(columns={col: "value"}))
        t.insert(2, "stratum", col)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def decision(summary: pd.DataFrame) -> str:
    orig = summary[summary.variant == "original"]
    rle = summary[summary.variant == "rle"]
    go_orig = orig[orig.go].path.tolist()
    go_rle = rle[rle.go].path.tolist()
    lines = [f"Pre-specified rule: GO if at least one conversion path materially changes "
             f">= {GO_THRESHOLD:.0%} of objects on the original COCO file."]
    if go_orig:
        lines.append("Result: GO. Paths over the threshold on original COCO: " + ", ".join(go_orig) + ".")
    elif go_rle:
        lines.append("Result: CONDITIONAL. No path crosses the threshold on original COCO; on the "
                     "RLE-stored variant these do: " + ", ".join(go_rle) + ". B would then rest on "
                     "datasets that store masks as RLE, which needs its own justification.")
    else:
        lines.append("Result: STOP or reframe. No path crosses the threshold on either variant.")
    lines.append("Material change = dropped, class changed, geometry IoU < 0.9, split into "
                 "several objects, or a crowd region written as an ordinary object.")
    return "\n".join(lines)


def _pct(k, n) -> float:
    return round(100 * float(k) / n, 2) if n else 0.0
