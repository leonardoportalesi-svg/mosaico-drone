"""Detección y matcheo de puntos en común (features) entre fotos.

Trabajamos a resolución reducida ("working") para que la alineación sea rápida
y liviana en memoria; el compositado final usa la resolución nativa.
"""
from __future__ import annotations

import cv2
import numpy as np

# --- detectores ---------------------------------------------------------------

def make_detector(name: str = "sift", max_features: int = 8000):
    name = name.lower()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=max_features)
    if name == "akaze":
        return cv2.AKAZE_create()
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    raise ValueError(f"detector desconocido: {name}")


def is_binary(name: str) -> bool:
    """ORB/AKAZE dan descriptores binarios; SIFT, flotantes."""
    return name.lower() in ("orb", "akaze")


# --- carga a resolución de trabajo -------------------------------------------

def load_working(path, work_megapix: float) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Lee la imagen y la reduce a ~work_megapix.

    Devuelve (bgr_work, gray_work, scale, (W_full, H_full)).
    scale es el factor full->work (<= 1).
    """
    bgr_full = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_full is None:
        raise IOError(f"no se pudo leer la imagen: {path}")
    h, w = bgr_full.shape[:2]
    if work_megapix > 0:
        scale = min(1.0, np.sqrt(work_megapix * 1e6 / (w * h)))
    else:
        scale = 1.0
    if scale < 1.0:
        bgr_work = cv2.resize(bgr_full, (round(w * scale), round(h * scale)),
                              interpolation=cv2.INTER_AREA)
    else:
        bgr_work = bgr_full
    gray = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2GRAY)
    return bgr_work, gray, scale, (w, h)


def detect(detector, gray: np.ndarray):
    """Detecta keypoints + descriptores en una imagen en gris."""
    kp, desc = detector.detectAndCompute(gray, None)
    return kp, desc


# --- matcheo ------------------------------------------------------------------

def make_matcher(detector_name: str):
    if is_binary(detector_name):
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    # FLANN KD-tree para descriptores flotantes (SIFT)
    index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def match(matcher, descA: np.ndarray, descB: np.ndarray, ratio: float = 0.75) -> list[tuple[int, int]]:
    """Matchea A->B con test de ratio de Lowe. Devuelve [(idxA, idxB), ...]."""
    if descA is None or descB is None or len(descA) < 2 or len(descB) < 2:
        return []
    if descA.dtype != np.float32 and not np.issubdtype(descA.dtype, np.integer):
        descA = descA.astype(np.float32)
        descB = descB.astype(np.float32)
    knn = matcher.knnMatch(descA, descB, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good
