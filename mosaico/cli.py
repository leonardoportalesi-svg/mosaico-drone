"""Interfaz de línea de comandos para Mosaico-Drone.

Ejemplo:
    python -m mosaico.cli ./fotos -o out/mosaico.tif
    python -m mosaico.cli a.jpg b.jpg c.jpg -o out/m.tif --gsd 0.04
"""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import MosaicOptions, build_mosaic


def _print_progress(phase, msg, i=None, n=None):
    bar = ""
    if i is not None and n:
        frac = i / n
        fill = int(20 * frac)
        bar = f" [{'█' * fill}{'·' * (20 - fill)}] {i}/{n}"
    sys.stderr.write(f"\r\033[K{phase:>12} │ {msg}{bar}")
    sys.stderr.flush()
    if phase == "listo":
        sys.stderr.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mosaico", description="Une fotos de drone georeferenciadas en un GeoTIFF.")
    ap.add_argument("inputs", nargs="+", help="carpeta(s) y/o archivos de imagen")
    ap.add_argument("-o", "--out", required=True, help="ruta del GeoTIFF de salida")
    ap.add_argument("--detector", default="sift", choices=["sift", "akaze", "orb"])
    ap.add_argument("--work-megapix", type=float, default=0.6,
                    help="resolución de alineación en Mpx (default 0.6)")
    ap.add_argument("--neighbors", type=int, default=10,
                    help="vecinos por GPS a matchear (default 10)")
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--world-model", default="affine", choices=["affine", "similarity"],
                    help="modelo de anclaje a UTM (default affine)")
    ap.add_argument("--gsd", type=float, default=None,
                    help="metros/píxel de salida (default: nativo)")
    ap.add_argument("--max-dim", type=int, default=30000,
                    help="lado máximo del mosaico en px (acota memoria)")
    ap.add_argument("--no-cog", action="store_true", help="GTiff normal en vez de COG")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true", help="imprime el reporte en JSON")
    args = ap.parse_args(argv)

    opts = MosaicOptions(
        detector=args.detector, work_megapix=args.work_megapix,
        neighbors=args.neighbors, min_inliers=args.min_inliers,
        world_model=args.world_model, gsd=args.gsd, max_dim=args.max_dim,
        cog=not args.no_cog,
    )
    cb = None if args.quiet else _print_progress
    try:
        report = build_mosaic(args.inputs, args.out, opts, cb)
    except Exception as e:
        sys.stderr.write(f"\nERROR: {e}\n")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        r = report
        print(f"\n✅ Mosaico: {r['out_path']}")
        print(f"   {r['n_placed']}/{r['n_input']} fotos unidas  ·  "
              f"EPSG:{r['epsg']}  ·  GSD {r['gsd']} m/px  ·  "
              f"{r['size_px'][0]}×{r['size_px'][1]} px")
        b = r["bounds_utm"]
        print(f"   Extensión UTM: E[{b['left']:.1f}, {b['right']:.1f}]  "
              f"N[{b['bottom']:.1f}, {b['top']:.1f}]")
        print(f"   Tiempo: {r['total_s']} s  ·  vista previa: {r['preview_path']}")
        for w in r["warnings"]:
            print(f"   ⚠️  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
