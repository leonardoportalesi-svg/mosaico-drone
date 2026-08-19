"""Compositado: warpea cada foto a resolución nativa y las funde en el mosaico.

Blending por feathering: el peso de cada foto es su distancia al borde (alto en
el centro, bajo en los bordes), así las costuras entre fotos solapadas se diluyen.
El acumulado se hace en float32 y al final se normaliza (Σ color·peso / Σ peso).
"""
from __future__ import annotations

import cv2
import numpy as np

from .model import Photo


def estimate_memory_gb(cols: int, rows: int) -> float:
    """RAM aproximada de los buffers de acumulación (sum RGB + pesos), en GB."""
    return (cols * rows * (3 + 1) * 4) / 1e9


def _out_bbox(p: Photo, cols: int, rows: int):
    """BBox entero (x0,y0,x1,y1) de la foto en el mosaico, recortado al canvas."""
    pts = p.full_corners()
    h = np.hstack([pts, np.ones((4, 1))])
    o = (p.H_out @ h.T).T
    o = o[:, :2] / o[:, 2:3]
    x0 = max(0, int(np.floor(o[:, 0].min())))
    y0 = max(0, int(np.floor(o[:, 1].min())))
    x1 = min(cols, int(np.ceil(o[:, 0].max())))
    y1 = min(rows, int(np.ceil(o[:, 1].max())))
    return x0, y0, x1, y1


def compose(photos: list[Photo], cols: int, rows: int, load_full, progress=None):
    """Funde las fotos en un RGBA (rows, cols, 4) uint8.

    load_full(photo) -> BGR full-res (se llama una vez por foto y se descarta).
    """
    accum = np.zeros((rows, cols, 3), np.float32)   # Σ color·peso (en RGB)
    wsum = np.zeros((rows, cols), np.float32)        # Σ peso
    placed = [p for p in photos if p.H_out is not None]

    for n, p in enumerate(placed):
        x0, y0, x1, y1 = _out_bbox(p, cols, rows)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        bgr = load_full(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # homografía trasladada para warpear directo al sub-rectángulo del canvas
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        Ht = T @ p.H_out
        warped = cv2.warpPerspective(rgb, Ht, (bw, bh), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT)
        ones = np.full(rgb.shape[:2], 255, np.uint8)
        mask = cv2.warpPerspective(ones, Ht, (bw, bh), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT)
        # peso = distancia al borde (feather). +1 para que el borde aporte algo.
        weight = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        weight = weight.astype(np.float32)

        sub = (slice(y0, y1), slice(x0, x1))
        accum[sub] += warped.astype(np.float32) * weight[..., None]
        wsum[sub] += weight
        if progress:
            progress(n + 1, len(placed))

    covered = wsum > 0
    rgb_out = np.zeros((rows, cols, 3), np.float32)
    np.divide(accum, wsum[..., None], out=rgb_out, where=covered[..., None])
    rgba = np.zeros((rows, cols, 4), np.uint8)
    rgba[..., :3] = np.clip(rgb_out, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(covered, 255, 0).astype(np.uint8)
    return rgba
