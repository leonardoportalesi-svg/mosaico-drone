"""Modelo de datos central: una Photo y su estado a lo largo del pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .exif_gps import GeoTag


@dataclass
class Photo:
    """Una foto y todo lo que se le va calculando en el pipeline.

    Convenciones de coordenadas:
      - full px  : píxeles de la imagen original (resolución nativa).
      - work px  : píxeles de la versión reducida usada para alinear.
      - comp px  : marco del componente (= work px de su imagen de referencia).
      - UTM      : metros en el CRS proyectado de salida.
      - out px   : píxeles del mosaico de salida.
    """

    path: Path
    index: int
    width: int                      # full px
    height: int
    geo: GeoTag | None = None

    # --- ingesta / georef de cámara ---
    utm: tuple[float, float] | None = None   # (easting, northing) del centro

    # --- alineación (working res) ---
    scale: float = 1.0              # factor full -> work (<= 1)
    work_size: tuple[int, int] = (0, 0)      # (ancho, alto) en work px
    kp: list = field(default_factory=list)   # keypoints (work px)
    desc: np.ndarray | None = None           # descriptores

    H_local: np.ndarray | None = None        # work px -> comp px (3x3)
    component: int = -1

    # --- georeferenciación / salida ---
    G: np.ndarray | None = None              # full px -> UTM (3x3)
    H_out: np.ndarray | None = None          # full px -> out px (3x3)

    @property
    def work_center(self) -> np.ndarray:
        w, h = self.work_size
        return np.array([w / 2.0, h / 2.0], np.float64)

    def full_corners(self) -> np.ndarray:
        """Las 4 esquinas en full px (orden TL, TR, BR, BL)."""
        w, h = self.width, self.height
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
