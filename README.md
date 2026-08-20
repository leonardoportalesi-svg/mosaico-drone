# Mosaico-Drone

Une fotos de drone georeferenciadas (JPG/PNG con GPS en el EXIF) en un **único
GeoTIFF georeferenciado, a resolución nativa**. Soltás las fotos, el programa
busca puntos en común entre ellas, las alinea y las pega en un solo mosaico
anclado a coordenadas reales (UTM).

🌐 **Probalo online:** https://mosaico-drone-production.up.railway.app

## Qué hace

```
Fotos JPG/PNG ─► Ingesta + GPS EXIF ─► Vecinos por GPS (KDTree)
                                              │
                                              ▼
GeoTIFF/COG ◄─ Georef + blending ◄─ Features (SIFT) + afín por par
 (UTM, GSD nativo)   (feather)        + bundle adjustment global
```

1. **Ingesta** — lee cada foto y extrae el GPS (lat/lon/alt) del EXIF; si es DJI, también el yaw del XMP.
2. **Vecinos por GPS** — proyecta a UTM y con un KD-tree busca solo los pares cercanos (evita comparar todas contra todas).
3. **Puntos en común** — SIFT + test de Lowe + afín por par con RANSAC, filtrando matcheos espurios.
4. **Alineación global** — *bundle adjustment* lineal robusto (IRLS): una transformación por foto resuelta con todas las aristas a la vez (sin drift), anclada al mundo con el GPS.
5. **Salida** — warpeo a resolución nativa + blending por *feathering* (tapa costuras) → GeoTIFF/COG en UTM, comprimido sin pérdida (DEFLATE) + overviews.

## Requisitos

- Python 3.13 (probado en macOS arm64).
- Las dependencias se instalan en un venv local (OpenCV, rasterio, pyproj, scipy, Flask). No hace falta GDAL del sistema: `rasterio` lo trae.

## Instalación

```bash
cd mosaico-drone
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Uso

### App web (drag & drop)

```bash
./start.sh                 # luego abrí http://127.0.0.1:8200
```

Soltás las fotos (o indicás la ruta de una carpeta del disco — recomendado para
muchas fotos) y le das **Generar mosaico**. Vas viendo el progreso y al final
tenés la vista previa + el botón para descargar el GeoTIFF.

### Línea de comandos

```bash
.venv/bin/python -m mosaico.cli ./mis_fotos -o out/mosaico.tif
.venv/bin/python -m mosaico.cli a.jpg b.jpg c.jpg -o out/m.tif --gsd 0.04 --no-cog
```

Opciones útiles:

| Flag | Default | Qué hace |
|------|---------|----------|
| `--gsd` | nativo | metros/píxel de salida (forzar resolución) |
| `--world-model` | `affine` | anclaje a UTM: `affine` o `similarity` |
| `--neighbors` | 10 | vecinos por GPS a matchear |
| `--min-inliers` | 20 | inliers mínimos para aceptar un par |
| `--detector` | `sift` | `sift`, `akaze` u `orb` |
| `--max-dim` | 30000 | lado máximo del mosaico en px (acota memoria) |
| `--no-cog` | — | escribe GTiff común en vez de COG |
| `--json` | — | imprime el reporte en JSON |

## Salida

GeoTIFF **RGBA** (la 4ª banda es alfa: transparente fuera del área cubierta),
en la zona **UTM/WGS84** correspondiente, en *tiles* con *overviews*. Por defecto
es un **COG** (Cloud-Optimized GeoTIFF). Abre directo en QGIS, GDAL, ArcGIS, etc.

## Datos de prueba

Sin fotos a mano, podés generar un set sintético con GPS real en el EXIF:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # agrega piexif
.venv/bin/python tools/make_synth.py --out tmp/synth --rows 4 --cols 5
.venv/bin/python -m mosaico.cli tmp/synth -o out/mosaico.tif
```

## Precisión y límites

- Es un **mosaico 2D** (no una ortofoto fotogramétrica completa). Funciona muy
  bien en terreno **plano-ish** y tomas **casi-nadir** (campos, lotes, urbano bajo).
  En edificios/árboles/relieve fuerte aparecen errores de paralaje.
- La precisión **relativa** (calidad del pegado) la dan los features y suele ser
  sub-píxel. La precisión **absoluta** (posición en el mundo) está limitada por el
  GPS de las fotos (~metros sin RTK/GCP).
- Las fotos que no logran unirse a ningún grupo (sin coincidencias o sin GPS
  suficiente en su componente) se reportan y se omiten, en vez de colocarse mal.
- Para entregables fotogramétricos de alta precisión (DSM, ortofoto métrica),
  considerá [OpenDroneMap](https://www.opendronemap.org/).

## Deploy público (Railway)

**App en vivo:** https://mosaico-drone-production.up.railway.app

La app corre bajo Docker + gunicorn (`Dockerfile`, `railway.json`). Pasos:

```bash
git init
git add -A
git commit -m "Mosaico-Drone"
gh repo create mosaico-drone --public --source=. --push   # o subilo a mano en GitHub
```

Después, en [railway.app](https://railway.app): **New Project → Deploy from GitHub repo**,
elegís el repo y Railway detecta el `Dockerfile` solo. No hace falta configurar
nada más — las variables de entorno de producción ya están en el `Dockerfile`.

**Diferencias del modo público vs. uso local:**
- La opción de "ruta de carpeta en el disco" se deshabilita (evita que cualquiera
  le pida al servidor que lea rutas arbitrarias). Solo queda subir fotos por drag&drop.
- Tope de subida: 300 MB / 400 fotos por vez (configurable con `MOSAICO_MAX_UPLOAD_MB`
  y `MOSAICO_MAX_FILES`).
- Los uploads y resultados de cada job se borran solos a las 6 h (`MOSAICO_MAX_AGE_HOURS`).
- Corre con **un solo worker** de gunicorn (el progreso de cada job vive en memoria
  de ese proceso) y varios threads, así que soporta varios usuarios a la vez pero
  no escala horizontalmente sin cambiar el manejo de estado de los jobs.

Si en algún momento accedés vos mismo al server por SSH/consola y querés la
comodidad de pasar rutas de disco, seteá `MOSAICO_ALLOW_LOCAL_PATHS=1` — pero
**no lo actives en un deploy expuesto a internet**, es la protección contra
path traversal.

## Estructura

```
mosaico/            # motor (paquete Python)
  exif_gps.py       # lectura de GPS desde EXIF/XMP
  features.py       # SIFT + matcheo
  align.py          # vecinos, afín por par, bundle adjustment global
  georef.py         # UTM, GSD, geotransform, escritura GeoTIFF/COG
  compose.py        # warpeo + blending por feathering
  pipeline.py       # orquestador
  cli.py            # interfaz de línea de comandos
webapp/             # app web (Flask + UI buildless)
tools/make_synth.py # generador de datos de prueba
Dockerfile          # imagen de producción (gunicorn)
railway.json        # config de deploy en Railway
```
