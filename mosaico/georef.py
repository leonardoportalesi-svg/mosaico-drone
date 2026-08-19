"""Georeferenciación de la salida: extensión UTM, GSD nativo, geotransform y escritura del GeoTIFF/COG."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

from .model import Photo


def _project_corners(H: np.ndarray, corners: np.ndarray) -> np.ndarray:
    pts = np.hstack([corners, np.ones((len(corners), 1))])
    out = (H @ pts.T).T
    return out[:, :2] / out[:, 2:3]


def _poly_area(c: np.ndarray) -> float:
    """Área del polígono (shoelace), c = Nx2."""
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def estimate_gsd(photos: list[Photo]) -> float:
    """GSD (metros/píxel) = mediana del tamaño de píxel nativo de cada foto.

    Se mide proyectando las 4 esquinas a UTM (con división proyectiva) y
    comparando el área del footprint en el suelo con el área en píxeles:
    GSD = sqrt(área_m2 / área_px). Robusto ante la componente proyectiva de G.
    """
    gsds = []
    for p in photos:
        if p.G is None:
            continue
        corners = _project_corners(p.G, p.full_corners())
        area_m2 = _poly_area(corners)
        area_px = p.width * p.height
        if area_m2 > 0 and area_px > 0:
            gsds.append(np.sqrt(area_m2 / area_px))
    if not gsds:
        raise ValueError("no hay fotos georeferenciadas para estimar el GSD")
    return float(np.median(gsds))


def world_bounds(photos: list[Photo]) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) en UTM, abarcando todas las fotos."""
    xs, ys = [], []
    for p in photos:
        if p.G is None:
            continue
        pts = _project_corners(p.G, p.full_corners())
        xs.extend(pts[:, 0])
        ys.extend(pts[:, 1])
    return min(xs), min(ys), max(xs), max(ys)


def build_output(photos: list[Photo], gsd: float, max_dim: int = 30000):
    """Arma transform de salida y fija H_out de cada foto.

    Si la dimensión supera max_dim, agranda el GSD (baja resolución) para acotar
    la memoria. Devuelve (affine, cols, rows, gsd_final).
    """
    left, bottom, right, top = world_bounds(photos)
    span_x, span_y = right - left, top - bottom

    cols = int(np.ceil(span_x / gsd))
    rows = int(np.ceil(span_y / gsd))
    if max(cols, rows) > max_dim:
        factor = max(cols, rows) / max_dim
        gsd *= factor
        cols = int(np.ceil(span_x / gsd))
        rows = int(np.ceil(span_y / gsd))

    # geotransform: top-left en (left, top), norte arriba
    affine = Affine(gsd, 0, left, 0, -gsd, top)
    # P: UTM -> out px  (inversa del geotransform, como homografía 3x3)
    P = np.array([[1 / gsd, 0, -left / gsd],
                  [0, -1 / gsd, top / gsd],
                  [0, 0, 1]])
    for p in photos:
        if p.G is not None:
            p.H_out = P @ p.G
    return affine, cols, rows, gsd


def write_geotiff(path, rgba: np.ndarray, affine: Affine, epsg: int,
                  cog: bool = True, compress: str = "DEFLATE"):
    """Escribe RGBA (H, W, 4) uint8 como GeoTIFF en tiles + overviews.

    Con cog=True produce un Cloud-Optimized GeoTIFF (driver COG, overviews
    automáticos); si no, un GTiff en tiles con overviews internos. La 4ª banda
    es alfa (transparencia fuera del área cubierta).
    """
    from rasterio.enums import ColorInterp
    h, w = rgba.shape[:2]
    bands = np.transpose(rgba, (2, 0, 1))  # (4, H, W)
    crs = CRS.from_epsg(epsg)
    ci = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]

    if cog:
        with rasterio.open(path, "w", driver="COG", width=w, height=h, count=4,
                           dtype="uint8", crs=crs, transform=affine,
                           compress=compress, blocksize=512,
                           overview_resampling="average") as dst:
            dst.write(bands)
            dst.colorinterp = ci
    else:
        with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=4,
                           dtype="uint8", crs=crs, transform=affine, tiled=True,
                           blockxsize=512, blockysize=512, compress=compress,
                           predictor=2) as dst:
            dst.write(bands)
            dst.colorinterp = ci
            dst.build_overviews(_overview_levels(w, h), Resampling.average)


def _overview_levels(w: int, h: int) -> list[int]:
    levels, f = [], 2
    while max(w, h) // f >= 256:
        levels.append(f)
        f *= 2
    return levels or [2]
