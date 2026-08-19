"""Genera un set sintético de "fotos de drone" con GPS en EXIF, para testear.

Renderiza un terreno procedural con mucha textura (features para SIFT), lo corta
en tiles solapados con rotación/jitter/variación de brillo, y le escribe a cada
tile su GPS real en el EXIF. Guarda truth.json con los límites UTM verdaderos
para poder verificar la georeferenciación del mosaico resultante.

Uso:
    python tools/make_synth.py --out tmp/synth --rows 4 --cols 5 --overlap 0.55
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def render_ground(w: int, h: int, seed: int) -> np.ndarray:
    """Imagen BGR con muchas features (campos, caminos, parcelas, manchas)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 60, np.uint8)
    # Fondo tipo "campos" con tonos verdosos/marrones por parcelas
    for _ in range(140):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        bw, bh = rng.integers(80, 360), rng.integers(80, 360)
        color = tuple(int(c) for c in rng.integers([20, 60, 20], [90, 170, 110]))
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), color, -1)
    # "Caminos" (líneas claras) para dar bordes fuertes
    for _ in range(18):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (180, 180, 185), int(rng.integers(2, 6)))
    # Detalles puntuales (techos, autos, árboles) = esquinas/blobs para SIFT
    for _ in range(900):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            r = int(rng.integers(2, 9))
            cv2.circle(img, (x, y), r, color, -1)
        else:
            s = int(rng.integers(4, 16))
            cv2.rectangle(img, (x, y), (x + s, y + s), color, -1)
    # Ruido fino para textura
    noise = rng.integers(-12, 12, (h, w, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def deg_to_dms_rationals(deg: float):
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60.0) * 3600.0
    return ((d, 1), (m, 1), (int(round(s * 10000)), 10000))


def write_jpg_with_gps(path: Path, bgr: np.ndarray, lat: float, lon: float, alt: float):
    """Guarda un BGR como JPEG con GPS en el EXIF (vía piexif)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    gps = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rationals(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rationals(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: (int(round(alt * 100)), 100),
    }
    exif_bytes = piexif.dump({"GPS": gps})
    im.save(path, "JPEG", quality=92, exif=exif_bytes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tmp/synth", help="carpeta de salida")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-w", type=int, default=800)
    ap.add_argument("--tile-h", type=int, default=600)
    ap.add_argument("--overlap", type=float, default=0.55, help="solape fraccional")
    ap.add_argument("--gsd", type=float, default=0.05, help="metros por píxel")
    ap.add_argument("--center-lat", type=float, default=-33.90)
    ap.add_argument("--center-lon", type=float, default=-60.57)
    ap.add_argument("--alt", type=float, default=120.0)
    ap.add_argument("--max-rot", type=float, default=6.0, help="rotación máx por tile (°)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tw, th = args.tile_w, args.tile_h
    step_x = int(tw * (1 - args.overlap))
    step_y = int(th * (1 - args.overlap))
    margin = max(tw, th)  # margen para que la rotación no se salga del terreno
    W = (args.cols - 1) * step_x + tw + 2 * margin
    H = (args.rows - 1) * step_y + th + 2 * margin
    ground = render_ground(W, H, args.seed)

    # CRS UTM correspondiente al centro elegido
    aoi = AreaOfInterest(args.center_lon - 0.1, args.center_lat - 0.1,
                         args.center_lon + 0.1, args.center_lat + 0.1)
    utm = query_utm_crs_info("WGS 84", aoi)[0]
    epsg = int(utm.code)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True)

    # Origen UTM tal que el centro del terreno caiga en (center_lat, center_lon)
    cx_e, cy_n = to_utm.transform(args.center_lon, args.center_lat)
    east0 = cx_e - (W / 2) * args.gsd   # UTM en el píxel (0,0) del terreno (top-left)
    north0 = cy_n + (H / 2) * args.gsd

    def px_to_utm(px, py):
        return east0 + px * args.gsd, north0 - py * args.gsd

    manifest = []
    idx = 0
    for r in range(args.rows):
        for c in range(args.cols):
            cx = margin + c * step_x + tw / 2
            cy = margin + r * step_y + th / 2
            cx += rng.uniform(-step_x * 0.06, step_x * 0.06)  # jitter de vuelo
            cy += rng.uniform(-step_y * 0.06, step_y * 0.06)
            ang = rng.uniform(-args.max_rot, args.max_rot)

            # Afín tile-local -> terreno: rota alrededor del centro del tile y traslada
            M = cv2.getRotationMatrix2D((tw / 2, th / 2), ang, 1.0)
            M[0, 2] += cx - tw / 2
            M[1, 2] += cy - th / 2
            tile = cv2.warpAffine(ground, M, (tw, th), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT)
            # Variación de brillo/exposición para testear el blending
            gain = rng.uniform(0.82, 1.18)
            tile = np.clip(tile.astype(np.float32) * gain, 0, 255).astype(np.uint8)

            e, n = px_to_utm(cx, cy)
            lon, lat = to_wgs.transform(e, n)
            name = f"DJI_{idx:04d}.jpg"
            write_jpg_with_gps(out / name, tile, lat, lon, args.alt)
            manifest.append({"name": name, "lat": lat, "lon": lon,
                             "utm_e": e, "utm_n": n, "ang": ang})
            idx += 1

    # Verdad-de-campo: extensión UTM real del terreno (top-left y bottom-right)
    e_min, n_max = px_to_utm(0, 0)
    e_max, n_min = px_to_utm(W, H)
    truth = {
        "epsg": epsg, "gsd": args.gsd, "ground_px": [W, H],
        "utm_bounds": {"left": e_min, "bottom": n_min, "right": e_max, "top": n_max},
        "center_latlon": [args.center_lat, args.center_lon],
        "n_tiles": idx, "tiles": manifest,
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"OK: {idx} fotos en {out}  |  CRS=EPSG:{epsg}  GSD={args.gsd} m/px")
    print(f"Extensión UTM verdadera: E[{e_min:.1f}, {e_max:.1f}]  N[{n_min:.1f}, {n_max:.1f}]")


if __name__ == "__main__":
    main()
