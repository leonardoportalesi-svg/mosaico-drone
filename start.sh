#!/usr/bin/env bash
# Lanza la app web de Mosaico-Drone en modo local. Uso: ./start.sh [puerto]  (default 8200)
# MOSAICO_ALLOW_LOCAL_PATHS=1: acá es seguro (es tu propia máquina) y habilita
# pasar la ruta de una carpeta del disco en vez de subir las fotos una por una.
cd "$(dirname "$0")" || exit 1
PORT="${1:-8200}"
echo "Abrí http://127.0.0.1:${PORT} en el navegador"
exec env MOSAICO_ALLOW_LOCAL_PATHS=1 .venv/bin/python webapp/server.py "$PORT"
