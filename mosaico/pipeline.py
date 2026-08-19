"""Orquestador del pipeline completo: de fotos a GeoTIFF georeferenciado."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import align, compose, features, georef
from .exif_gps import image_size, read_geotags
from .model import Photo

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class MosaicOptions:
    detector: str = "sift"
    work_megapix: float = 0.6        # resolución de alineación (~0.6 Mpx)
    max_features: int = 8000
    ratio: float = 0.75              # test de Lowe
    neighbors: int = 10              # vecinos por GPS a matchear
    ransac_thresh: float = 4.0       # px (afín por par)
    min_inliers: int = 20
    world_model: str = "affine"      # ref->UTM: "affine" o "similarity"
    gsd: float | None = None         # m/px de salida (None = nativo, mediana)
    max_dim: int = 30000             # tope de lado del mosaico (acota memoria)
    cog: bool = True
    compress: str = "DEFLATE"
    preview_max: int = 1600          # lado máx del PNG de vista previa


def _noop(*_a, **_k):
    pass


def discover_images(inputs) -> list[Path]:
    """Acepta archivos y/o carpetas; devuelve las imágenes encontradas."""
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += [q for q in sorted(p.iterdir()) if q.suffix.lower() in IMG_EXTS]
        elif p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def build_mosaic(inputs, out_path, opts: MosaicOptions | None = None, cb=None) -> dict:
    opts = opts or MosaicOptions()
    cb = cb or _noop
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def stamp(name, since):
        timings[name] = round(time.perf_counter() - since, 2)

    # 1. ingesta -------------------------------------------------------------
    cb("ingesta", "Buscando imágenes…", 0, 1)
    paths = discover_images(inputs)
    if len(paths) < 2:
        raise ValueError("se necesitan al menos 2 imágenes")
    photos: list[Photo] = []
    for i, p in enumerate(paths):
        try:
            w, h = image_size(p)
        except Exception:
            continue
        photos.append(Photo(path=p, index=len(photos), width=w, height=h,
                            geo=read_geotags(p)))
        cb("ingesta", f"Leyendo EXIF {i + 1}/{len(paths)}", i + 1, len(paths))
    n_with_gps = sum(p.geo is not None for p in photos)
    if n_with_gps < 2:
        raise ValueError(f"solo {n_with_gps} foto(s) con GPS; se necesitan ≥2")

    # 2. GPS -> UTM ----------------------------------------------------------
    epsg, _ = align.assign_utm(photos)

    # 3. features ------------------------------------------------------------
    t = time.perf_counter()
    detector = features.make_detector(opts.detector, opts.max_features)
    total_feats = 0
    for i, p in enumerate(photos):
        try:
            _, gray, scale, (w, h) = features.load_working(p.path, opts.work_megapix)
        except Exception:
            continue
        p.scale = scale
        p.work_size = (gray.shape[1], gray.shape[0])
        p.kp, p.desc = features.detect(detector, gray)
        total_feats += 0 if p.desc is None else len(p.desc)
        cb("features", f"Detectando puntos {i + 1}/{len(photos)}", i + 1, len(photos))
    stamp("features", t)

    # 4. vecinos + grafo -----------------------------------------------------
    t = time.perf_counter()
    pairs = align.neighbor_pairs(photos, opts.neighbors)
    matcher = features.make_matcher(opts.detector)

    def graph_progress(done, total, edges):
        cb("matcheo", f"Matcheando pares {done}/{total} ({edges} con coincidencias)",
           done, total)

    edges, adj, uf = align.build_graph(photos, pairs, matcher, opts.ratio,
                                       opts.ransac_thresh, opts.min_inliers,
                                       graph_progress)
    stamp("matcheo", t)

    # 5. componentes + bundle adjustment global -----------------------------
    comps = align.components(photos, uf)
    placed, dropped = align.global_adjust(photos, edges, adj, comps, opts.world_model)
    georeferenced = [p for p in photos if p.G is not None]
    if not georeferenced:
        raise ValueError("no se pudo georeferenciar ningún grupo de fotos "
                         "(¿faltan coincidencias o GPS?)")

    # 6. georeferenciación de salida ----------------------------------------
    gsd = opts.gsd or georef.estimate_gsd(georeferenced)
    affine, cols, rows, gsd = georef.build_output(georeferenced, gsd, opts.max_dim)
    mem_gb = compose.estimate_memory_gb(cols, rows)
    cb("georef", f"Mosaico {cols}×{rows} px @ {gsd:.4f} m/px (~{mem_gb:.1f} GB RAM)",
       1, 1)

    # 7. compositado ---------------------------------------------------------
    t = time.perf_counter()

    def load_full(p):
        return cv2.imread(str(p.path), cv2.IMREAD_COLOR)

    def comp_progress(done, total):
        cb("compositado", f"Fusionando fotos {done}/{total}", done, total)

    rgba = compose.compose(georeferenced, cols, rows, load_full, comp_progress)
    stamp("compositado", t)

    # 8. escritura -----------------------------------------------------------
    t = time.perf_counter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    georef.write_geotiff(out_path, rgba, affine, epsg, opts.cog, opts.compress)
    preview_path = out_path.with_suffix(".preview.png")
    _save_preview(rgba, preview_path, opts.preview_max)
    stamp("escritura", t)

    left, bottom, right, top = georef.world_bounds(georeferenced)
    report = {
        "out_path": str(out_path),
        "preview_path": str(preview_path),
        "n_input": len(photos),
        "n_with_gps": n_with_gps,
        "n_features_total": total_feats,
        "n_pairs_tested": len(pairs),
        "n_edges": len(edges) // 2,
        "n_components": len(comps),
        "n_placed": len(georeferenced),
        "n_dropped": len(dropped),
        "dropped_paths": [photos[i].path.name for i in dropped],
        "epsg": epsg,
        "gsd": round(gsd, 5),
        "size_px": [cols, rows],
        "bounds_utm": {"left": left, "bottom": bottom, "right": right, "top": top},
        "mem_gb": round(mem_gb, 2),
        "cog": opts.cog,
        "timings_s": timings,
        "total_s": round(time.perf_counter() - t0, 2),
        "warnings": _warnings(dropped, photos),
    }
    cb("listo", f"Mosaico generado: {out_path.name}", 1, 1)
    return report


def _warnings(dropped, photos) -> list[str]:
    w = []
    if dropped:
        w.append(f"{len(dropped)} foto(s) no se pudieron unir al mosaico "
                 "(sin suficientes coincidencias o GPS en su grupo).")
    return w


def _save_preview(rgba: np.ndarray, path: Path, max_side: int):
    h, w = rgba.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = rgba if s >= 1.0 else cv2.resize(rgba, (round(w * s), round(h * s)),
                                           interpolation=cv2.INTER_AREA)
    bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)
