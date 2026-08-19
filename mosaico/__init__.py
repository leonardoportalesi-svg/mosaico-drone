"""Mosaico-Drone: une fotos de drone georeferenciadas en un único GeoTIFF.

Pipeline:
    ingesta (EXIF/GPS) -> vecinos por GPS -> features+homografías ->
    alineación global -> georeferenciación (UTM) -> compositado -> GeoTIFF/COG
"""

__version__ = "0.1.0"
