"""Lectura de geo-etiquetas (GPS) desde EXIF y XMP de fotos de drone.

Las fotos de drone traen la posición de la cámara en el EXIF estándar
(GPSLatitude/Longitude/Altitude). Los DJI además guardan en XMP la altura
relativa de vuelo y el yaw del gimbal, que usamos como pistas opcionales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PIL.ExifTags import IFD

# Tags dentro del IFD de GPS (ver EXIF spec)
_GPS_LAT_REF, _GPS_LAT = 1, 2
_GPS_LON_REF, _GPS_LON = 3, 4
_GPS_ALT_REF, _GPS_ALT = 5, 6

# Atributos XMP de DJI (best-effort vía regex sobre el packet XMP crudo)
_RE_YAW = re.compile(rb'(?:GimbalYawDegree|FlightYawDegree)="([-+]?\d+(?:\.\d+)?)"')
_RE_RELALT = re.compile(rb'RelativeAltitude="([-+]?\d+(?:\.\d+)?)"')


@dataclass
class GeoTag:
    """Geo-referencia de una foto, en grados decimales WGS84."""

    lat: float
    lon: float
    alt: float | None = None       # altitud absoluta (GPSAltitude), metros
    rel_alt: float | None = None    # altura relativa de vuelo (XMP DJI), metros
    yaw: float | None = None        # rumbo de cámara, grados (0=N, sentido horario)


def _to_float(x) -> float:
    """Convierte un IFDRational / tupla (num, den) / número a float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return x[0] / x[1]
        except Exception:  # pragma: no cover - formato inesperado
            return float(x.numerator) / float(x.denominator)


def _dms_to_deg(dms, ref) -> float:
    """(grados, minutos, segundos) + hemisferio -> grados decimales con signo."""
    d, m, s = (_to_float(v) for v in dms)
    val = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W", b"S", b"W"):
        val = -val
    return val


def _read_xmp_hints(path: Path) -> tuple[float | None, float | None]:
    """Devuelve (rel_alt, yaw) leyendo el packet XMP crudo si existe."""
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    start = data.find(b"<x:xmpmeta")
    if start == -1:
        return None, None
    end = data.find(b"</x:xmpmeta>", start)
    packet = data[start : end + 12] if end != -1 else data[start : start + 20000]
    rel_alt = float(m.group(1)) if (m := _RE_RELALT.search(packet)) else None
    yaw = float(m.group(1)) if (m := _RE_YAW.search(packet)) else None
    return rel_alt, yaw


def read_geotags(path) -> GeoTag | None:
    """Lee la geo-etiqueta de una imagen. Devuelve None si no tiene GPS."""
    path = Path(path)
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            gps = exif.get_ifd(IFD.GPSInfo)
    except Exception:
        return None

    if not gps or _GPS_LAT not in gps or _GPS_LON not in gps:
        return None

    try:
        lat = _dms_to_deg(gps[_GPS_LAT], gps.get(_GPS_LAT_REF, "N"))
        lon = _dms_to_deg(gps[_GPS_LON], gps.get(_GPS_LON_REF, "E"))
    except Exception:
        return None

    alt = None
    if _GPS_ALT in gps:
        alt = _to_float(gps[_GPS_ALT])
        if gps.get(_GPS_ALT_REF, 0) in (1, b"\x01"):  # 1 = bajo el nivel del mar
            alt = -alt

    rel_alt, yaw = _read_xmp_hints(path)
    return GeoTag(lat=lat, lon=lon, alt=alt, rel_alt=rel_alt, yaw=yaw)


def image_size(path) -> tuple[int, int]:
    """(ancho, alto) en píxeles sin decodificar los datos de la imagen."""
    with Image.open(path) as img:
        return img.size
