# Mosaico-Drone — imagen de producción (Railway/Render/Fly.io/cualquier host Docker)
FROM python:3.13-slim

# libgomp1: requerida por OpenCV (paralelismo interno).
# rasterio trae su propio GDAL embebido en el wheel, no hace falta instalarlo aparte.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY mosaico/ mosaico/
COPY webapp/ webapp/

# Deploy público por default: sin rutas de disco arbitrarias, con tope de subida.
ENV MOSAICO_ALLOW_LOCAL_PATHS=0 \
    MOSAICO_MAX_UPLOAD_MB=300 \
    MOSAICO_MAX_FILES=400 \
    MOSAICO_MAX_AGE_HOURS=6 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# workers=1: el progreso de cada job vive en memoria de un solo proceso (dict JOBS);
# con más workers una request podría caer en un proceso que no conoce ese job_id.
# threads>1: para atender la barra de progreso (SSE) mientras corre el pipeline en background.
# timeout=0: el pipeline puede tardar minutos con muchas fotos; no hay que cortar la request.
CMD gunicorn webapp.server:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --access-logfile - --error-logfile -
