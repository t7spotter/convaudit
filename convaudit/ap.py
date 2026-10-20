"""COCO AP from pycocotools' per-image results, with image weights for bootstrap.

COCOeval.evaluate() does the per-image matching; this module replaces only
the accumulate step. With all weights 1 it reproduces COCOeval's AP exactly.
A weight of k for an image is the same as that image appearing k times, so a
bootstrap resample of images is a vector of counts, and AP is recomputed
without re-running the matching.
"""

from __future__ import annotations

import numpy as np
from pycocotools.cocoeval import COCOeval

AREAS = ("all", "small", "medium", "large")


class APTable:
    """Matching results of one COCOeval run, flattened per category and area."""

    def __init__(self, E: COCOeval, max_det: int = 100):
        p = E.params
        self.img_ids = list(p.imgIds)
        self.n_img = len(self.img_ids)
        self.rec_thrs = np.asarray(p.recThrs)
        self.n_thr = len(p.iouThrs)
        A = len(p.areaRng)
        I = self.n_img
        self.groups = {}  # (area_index, category_index) -> arrays sorted by score
        for k in range(len(p.catIds)):
            for a in range(A):
                es = [E.evalImgs[k * A * I + a * I + i] for i in range(I)]
                scores, img_idx, tp, fp, npig = [], [], [], [], np.zeros(I)
                for i, e in enumerate(es):
                    if e is None:
                        continue
                    s = np.asarray(e["dtScores"][:max_det])
                    dtm = np.asarray(e["dtMatches"])[:, :max_det]
                    dti = np.asarray(e["dtIgnore"])[:, :max_det]
                    npig[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
                    if s.size:
                        scores.append(s)
                        img_idx.append(np.full(s.size, i))
                        tp.append(np.logical_and(dtm != 0, np.logical_not(dti)))
                        fp.append(np.logical_and(dtm == 0, np.logical_not(dti)))
                if not npig.any():
                    continue  # COCOeval leaves this category at -1 and skips it
                if scores:
                    s = np.concatenate(scores)
                    order = np.argsort(-s, kind="mergesort")
                    arr = (np.concatenate(img_idx)[order],
                           np.concatenate(tp, axis=1)[:, order].astype(float),
                           np.concatenate(fp, axis=1)[:, order].astype(float))
                else:
                    arr = (np.zeros(0, int), np.zeros((self.n_thr, 0)), np.zeros((self.n_thr, 0)))
                self.groups[(a, k)] = (*arr, npig)

    def ap(self, weights: np.ndarray | None = None, area: str = "all",
           iou: float | None = None) -> float:
        """AP x 100. iou=None averages IoU 0.50:0.95; 0.5 or 0.75 picks one."""
        a = AREAS.index(area)
        w_img = np.ones(self.n_img) if weights is None else np.asarray(weights, float)
        thr = slice(None) if iou is None else [int(round((iou - 0.5) / 0.05))]
        vals = []
        for (aa, _), (img, tp, fp, npig) in self.groups.items():
            if aa != a:
                continue
            n_pos = float(np.dot(npig, w_img))
            if n_pos == 0:
                continue
            w = w_img[img]
            tps = np.cumsum(tp[thr] * w, axis=1)
            fps = np.cumsum(fp[thr] * w, axis=1)
            for t in range(tps.shape[0]):
                vals.append(_interp_ap(tps[t], fps[t], n_pos, self.rec_thrs))
        return 100.0 * float(np.mean(vals)) if vals else float("nan")


def _interp_ap(tp: np.ndarray, fp: np.ndarray, n_pos: float, rec_thrs: np.ndarray) -> float:
    if tp.size == 0:
        return 0.0
    keep = np.ones(tp.size, bool)  # drop zero-weight steps (images not drawn)
    keep[1:] = (np.diff(tp) + np.diff(fp)) > 0
    keep[0] = (tp[0] + fp[0]) > 0
    tp, fp = tp[keep], fp[keep]
    if tp.size == 0:
        return 0.0
    rc = tp / n_pos
    pr = tp / np.maximum(tp + fp, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]  # precision envelope
    idx = np.searchsorted(rc, rec_thrs, side="left")
    q = np.where(idx < pr.size, pr[np.minimum(idx, pr.size - 1)], 0.0)
    return float(q.mean())


def run_cocoeval(gt, dt, iou_type: str) -> COCOeval:
    E = COCOeval(gt, dt, iou_type)
    E.params.imgIds = sorted(gt.getImgIds())
    E.evaluate()
    return E


def bootstrap_weights(n_img: int, n: int, seed: int = 0) -> np.ndarray:
    """n resamples of n_img images with replacement, as per-image counts."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
                     for _ in range(n)]).astype(float)
