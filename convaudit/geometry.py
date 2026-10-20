"""Geometry helpers. All rasterization uses pycocotools, the COCO reference."""

from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        if not seg:
            return np.zeros((h, w), np.uint8)
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


def polygons_to_mask(polys: list[list[float]], h: int, w: int) -> np.ndarray:
    polys = [p for p in polys if len(p) >= 6]
    if not polys:
        return np.zeros((h, w), np.uint8)
    rle = mask_utils.merge(mask_utils.frPyObjects(polys, h, w))
    return mask_utils.decode(rle).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_ioa(inner: np.ndarray, outer: np.ndarray) -> float:
    """Share of `inner`'s area that lies inside `outer` (xyxy)."""
    tl = np.maximum(inner[:2], outer[:2])
    br = np.minimum(inner[2:], outer[2:])
    inter = float(np.prod(np.clip(br - tl, 0, None)))
    area = float(np.prod(np.clip(inner[2:] - inner[:2], 0, None)))
    return inter / area if area > 0 else 0.0


def poly_extent(poly: list[float]) -> list[float]:
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_axis_rectangle(poly: list[float], box: list[float], tol: float = 0.5) -> bool:
    """True if a 4-point polygon is just the corners of `box`."""
    if len(poly) != 8:
        return False
    pts = np.asarray(poly).reshape(4, 2)
    xs, ys = sorted(set(np.round(pts[:, 0], 3))), sorted(set(np.round(pts[:, 1], 3)))
    if len(xs) != 2 or len(ys) != 2:
        return False
    return (abs(xs[0] - box[0]) <= tol and abs(xs[1] - box[2]) <= tol
            and abs(ys[0] - box[1]) <= tol and abs(ys[1] - box[3]) <= tol)


def mask_components(m: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    return int(n - 1)


def mask_has_holes(m: np.ndarray) -> bool:
    if not m.any():
        return False
    contours, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    return bool((hier[0][:, 3] >= 0).any())  # any contour with a parent is a hole
