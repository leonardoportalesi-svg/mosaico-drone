"""Alineación: de fotos sueltas a un marco común georeferenciado.

Pasos:
  1. assign_utm     -> proyecta el GPS de cada foto a UTM.
  2. neighbor_pairs -> pares candidatos (vecinos por GPS, evita O(n^2)).
  3. build_graph    -> transformación afín por pares (RANSAC) -> grafo.
  4. components     -> componentes conexas del grafo.
  5. global_adjust  -> bundle adjustment lineal: una similitud foto->mundo por
                       foto, resuelta con TODAS las aristas a la vez (los cierres
                       de lazo eliminan el drift) y el GPS como anclaje absoluto.
"""
from __future__ import annotations

import cv2
import numpy as np
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from . import features as F
from .model import Photo


# --- 1. GPS -> UTM ------------------------------------------------------------

def assign_utm(photos: list[Photo]) -> tuple[int, Transformer]:
    """Elige la zona UTM por el centroide y proyecta el GPS de cada foto."""
    geos = [p.geo for p in photos if p.geo is not None]
    if not geos:
        raise ValueError("ninguna foto tiene GPS en el EXIF")
    lats = [g.lat for g in geos]
    lons = [g.lon for g in geos]
    aoi = AreaOfInterest(min(lons), min(lats), max(lons), max(lats))
    info = query_utm_crs_info("WGS 84", aoi)
    if not info:
        raise ValueError("no se pudo determinar la zona UTM")
    epsg = int(info[0].code)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    for p in photos:
        if p.geo is not None:
            e, n = to_utm.transform(p.geo.lon, p.geo.lat)
            p.utm = (e, n)
    return epsg, to_utm


# --- 2. vecinos por GPS -------------------------------------------------------

def neighbor_pairs(photos: list[Photo], k: int = 6) -> list[tuple[int, int]]:
    """Pares candidatos: cada foto con sus k vecinos más cercanos en UTM."""
    idx = [p.index for p in photos if p.utm is not None]
    if len(idx) < 2:
        return []
    coords = np.array([photos[i].utm for i in idx])
    tree = cKDTree(coords)
    kk = min(k + 1, len(idx))
    _, nbrs = tree.query(coords, k=kk)
    pairs = set()
    for a_local, row in enumerate(np.atleast_2d(nbrs)):
        a = idx[a_local]
        for b_local in np.atleast_1d(row):
            b = idx[int(b_local)]
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


# --- 3. transformación por par + grafo ---------------------------------------

def pairwise_transform(pa: Photo, pb: Photo, matcher, ratio: float,
                       ransac_thresh: float, min_inliers: int):
    """Afín que lleva puntos de pa -> pb (work px). Devuelve (A_3x3, n_inliers) o None.

    Afín (en vez de homografía) es lo apropiado para tomas casi-nadir: es estable,
    extrapola sin reventar y no inventa términos proyectivos por ruido.
    """
    m = F.match(matcher, pa.desc, pb.desc, ratio)
    if len(m) < max(8, min_inliers):
        return None
    src = np.float32([pa.kp[i].pt for i, _ in m])
    dst = np.float32([pb.kp[j].pt for _, j in m])
    M, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                   ransacReprojThreshold=ransac_thresh,
                                   maxIters=4000, confidence=0.999)
    if M is None or mask is None:
        return None
    inliers = int(mask.sum())
    if inliers < min_inliers:
        return None
    if not _plausible_affine(M[:, :2]):  # descarta matcheos espurios baratos
        return None
    return np.vstack([M, [0, 0, 1.0]]), inliers


def _plausible_affine(L: np.ndarray, scale_lo=0.5, scale_hi=2.0, max_aniso=1.5) -> bool:
    """¿La parte lineal de la afín es geométricamente razonable entre vecinas?

    Fotos vecinas (GPS cercano) están a altitud similar: escala ~1, sin reflexión
    y poco corte. Una afín muy escalada/anisótropa/reflejada delata un falso match.
    """
    det = np.linalg.det(L)
    if det <= 0:  # reflexión -> match imposible
        return False
    if not (scale_lo <= np.sqrt(det) <= scale_hi):
        return False
    sv = np.linalg.svd(L, compute_uv=False)
    return sv[0] / max(sv[1], 1e-9) <= max_aniso


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_graph(photos, pairs, matcher, ratio, ransac_thresh, min_inliers, progress=None):
    """Devuelve (edges, adj, uf). edges[(i,j)] guarda la afín i->j (i<j)."""
    edges: dict[tuple[int, int], np.ndarray] = {}
    adj: dict[int, list[tuple[int, int]]] = {p.index: [] for p in photos}
    uf = _UnionFind(len(photos))
    for n, (a, b) in enumerate(pairs):
        res = pairwise_transform(photos[a], photos[b], matcher, ratio,
                                 ransac_thresh, min_inliers)
        if res is not None:
            A_ab, inl = res
            edges[(a, b)] = A_ab
            adj[a].append((b, inl))
            adj[b].append((a, inl))
            uf.union(a, b)
        if progress:
            progress(n + 1, len(pairs), len(edges))
    return edges, adj, uf


def components(photos, uf) -> dict[int, list[int]]:
    comps: dict[int, list[int]] = {}
    for p in photos:
        comps.setdefault(uf.find(p.index), []).append(p.index)
        p.component = uf.find(p.index)
    return comps


# --- 4/5. bundle adjustment lineal global ------------------------------------

def _samples(w: int, h: int) -> np.ndarray:
    """Puntos de muestreo en el marco de una foto (esquinas + centro), work px."""
    return np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], np.float64)


def global_adjust(photos: list[Photo], edges, adj, comps, world_model: str = "affine"):
    """Georeferencia cada componente en dos etapas (features = relativo, GPS = absoluto).

      1) BA relativo: fija una foto de referencia y resuelve la estructura
         relativa (similitud foto->ref) usando TODAS las aristas (sin drift),
         en píxeles (bien condicionado, sin GPS).
      2) Anclaje: ajusta ref->UTM con el GPS (afín/similitud, RANSAC) y compone
         G = T · Sim · S  (full px -> UTM).

    Solo se resuelven componentes con >=2 fotos con GPS. Devuelve (placed, dropped).
    """
    placed, dropped = [], []
    for nodes in comps.values():
        gps_nodes = [i for i in nodes if photos[i].utm is not None]
        if len(nodes) < 2 or len(gps_nodes) < 2:
            dropped.extend(nodes)
            continue
        sims = _relative_ba(photos, nodes, edges, adj)
        if sims is None:
            dropped.extend(nodes)
            continue
        # ref-frame px <-> UTM en los centros con GPS
        ref_pts = np.array([_apply_h(sims[i], photos[i].work_center) for i in gps_nodes])
        utm_pts = np.array([photos[i].utm for i in gps_nodes])
        T = _fit_world(ref_pts, utm_pts, world_model)
        if T is None:
            dropped.extend(nodes)
            continue
        for i in nodes:
            S = np.diag([photos[i].scale, photos[i].scale, 1.0])  # full -> work
            photos[i].G = T @ sims[i] @ S
        placed.extend(nodes)
    return placed, dropped


def _relative_ba(photos, nodes, edges, adj, iters: int = 5, huber_px: float = 3.0):
    """Etapa 1: similitud foto->marco-de-referencia, robusta a aristas malas.

    Fija la foto más conectada como referencia (identidad) para anclar el gauge
    sin GPS, y resuelve por mínimos cuadrados reponderados (IRLS, kernel de
    Huber): las aristas con residuo alto (matcheos espurios) se van anulando.
    Devuelve {idx: Sim 3x3} o None.
    """
    ref = max(nodes, key=lambda i: len(adj[i]))
    unknown = [i for i in nodes if i != ref]
    if not unknown:
        return {ref: np.eye(3)}
    block = {idx: 4 * b for b, idx in enumerate(unknown)}

    def terms_x(k, x, y):
        if k == ref:
            return [], x
        b = block[k]
        return [(b, x), (b + 1, -y), (b + 2, 1)], 0.0

    def terms_y(k, x, y):
        if k == ref:
            return [], y
        b = block[k]
        return [(b, y), (b + 1, x), (b + 3, 1)], 0.0

    rows, cols, vals, rhs, erow = [], [], [], [], []
    eq = [0]

    def add(eid, ci, consti, cj, constj):
        r = eq[0]
        for col, v in ci:
            rows.append(r); cols.append(col); vals.append(v)
        for col, v in cj:
            rows.append(r); cols.append(col); vals.append(-v)
        rhs.append(constj - consti)  # world_i - world_j = 0
        erow.append(eid)
        eq[0] += 1

    eid = {}
    for (i, j), A_ij in edges.items():
        if (i not in block and i != ref) or (j not in block and j != ref):
            continue
        e = eid.setdefault((i, j), len(eid))
        for x, y in _samples(*photos[i].work_size):
            t = A_ij @ np.array([x, y, 1.0])
            tx, ty = t[0], t[1]
            cix, kix = terms_x(i, x, y); cjx, kjx = terms_x(j, tx, ty)
            add(e, cix, kix, cjx, kjx)
            ciy, kiy = terms_y(i, x, y); cjy, kjy = terms_y(j, tx, ty)
            add(e, ciy, kiy, cjy, kjy)

    if eq[0] < 4 * len(unknown) or not eid:
        return None
    A = coo_matrix((vals, (rows, cols)), shape=(eq[0], 4 * len(unknown))).tocsr()
    rhs = np.array(rhs)
    erow = np.array(erow)
    counts = np.bincount(erow, minlength=len(eid))
    w_edge = np.ones(len(eid))

    sol = None
    for _ in range(iters):
        wr = np.sqrt(w_edge[erow])
        sol = lsqr(diags(wr) @ A, rhs * wr, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
        resid = A @ sol - rhs
        # residuo RMS por arista (cada eq aporta una componente x o y, en px)
        sse = np.bincount(erow, weights=resid ** 2, minlength=len(eid))
        r_edge = np.sqrt(sse / np.maximum(counts, 1))
        w_edge = np.where(r_edge <= huber_px, 1.0, huber_px / np.maximum(r_edge, 1e-9))

    sims = {ref: np.eye(3)}
    for idx, b in block.items():
        a, bb, tx, ty = sol[b:b + 4]
        sims[idx] = np.array([[a, -bb, tx], [bb, a, ty], [0, 0, 1.0]])
    return sims


def _fit_world(src: np.ndarray, dst: np.ndarray, model: str):
    """Etapa 2: ajusta src(ref px) -> dst(UTM). 'affine' o 'similarity'. 3x3 o None."""
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    use_affine = model == "affine" and len(src) >= 4 and not _collinear(src)
    if use_affine:
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    else:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=8.0)
    if M is None:
        return None
    return np.vstack([M, [0, 0, 1.0]])


def _collinear(pts: np.ndarray, tol: float = 1e-3) -> bool:
    c = pts - pts.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return s[-1] / (s[0] + 1e-12) < tol


def _apply_h(H: np.ndarray, pt) -> np.ndarray:
    v = H @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]
