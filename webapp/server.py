"""Servidor web local de Mosaico-Drone (Flask).

Sirve la UI buildless de drag & drop y expone una API con progreso en vivo (SSE).
Para datasets grandes conviene pasar la ruta de una carpeta (el servidor lee del
disco local) en vez de subir las fotos por el navegador.

Uso:
    python webapp/server.py            # luego abrir http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, request, send_file  # noqa: E402

from mosaico.pipeline import MosaicOptions, build_mosaic  # noqa: E402

# En un despliegue público NO hay que confiar en una ruta de disco tipeada por
# el visitante (path traversal / lectura arbitraria de archivos). Por default
# el server exige que el "folder" venga de /api/upload (dentro de UPLOAD_DIR).
# Para uso local (./start.sh) se habilita la comodidad de pasar cualquier ruta.
ALLOW_LOCAL_PATHS = os.environ.get("MOSAICO_ALLOW_LOCAL_PATHS", "0") == "1"
MAX_UPLOAD_MB = int(os.environ.get("MOSAICO_MAX_UPLOAD_MB", "300"))
MAX_FILES = int(os.environ.get("MOSAICO_MAX_FILES", "400"))

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = None if ALLOW_LOCAL_PATHS else MAX_UPLOAD_MB * 1024 * 1024

OUT_DIR = PROJECT_ROOT / "out" / "web"
UPLOAD_DIR = OUT_DIR / "uploads"
OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}
MAX_AGE_H = int(os.environ.get("MOSAICO_MAX_AGE_HOURS", "6"))


def _cleanup_old_dirs():
    """Borra uploads/jobs de más de MAX_AGE_H horas (evita llenar el disco)."""
    import shutil
    import time
    cutoff = time.time() - MAX_AGE_H * 3600
    for base in (UPLOAD_DIR, OUT_DIR / "jobs"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass


def _safe_folder(raw: str) -> Path | None:
    """Valida la carpeta pedida por el cliente.

    En modo público, solo se aceptan carpetas creadas por /api/upload (dentro
    de UPLOAD_DIR); cualquier otra ruta se rechaza. En modo local se permite
    cualquier carpeta existente, para poder apuntar a fotos ya en el disco.
    """
    if not raw:
        return None
    p = Path(raw).resolve()
    if not p.is_dir():
        return None
    if ALLOW_LOCAL_PATHS:
        return p
    try:
        p.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        return None
    return p


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/config")
def config():
    return jsonify(allow_local_paths=ALLOW_LOCAL_PATHS, max_upload_mb=MAX_UPLOAD_MB,
                   max_files=MAX_FILES)


@app.post("/api/upload")
def upload():
    """Recibe fotos por multipart y las guarda en una carpeta temporal."""
    _cleanup_old_dirs()
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="no se recibieron archivos"), 400
    if len(files) > MAX_FILES:
        return jsonify(error=f"máximo {MAX_FILES} fotos por vez"), 400
    folder = UPLOAD_DIR / uuid.uuid4().hex
    folder.mkdir(parents=True)
    for f in files:
        name = Path(f.filename).name  # evita rutas relativas maliciosas
        if name:
            f.save(folder / name)
    return jsonify(folder=str(folder), count=len(files))


@app.post("/api/start")
def start():
    data = request.get_json(force=True)
    raw_folder = (data.get("folder") or "").strip()
    folder = _safe_folder(raw_folder)
    if folder is None:
        return jsonify(error="carpeta no válida (subí las fotos primero)"), 400
    folder = str(folder)

    opts = MosaicOptions(
        world_model=data.get("world_model", "affine"),
        cog=bool(data.get("cog", True)),
        max_dim=int(data.get("max_dim", 30000)),
        gsd=float(data["gsd"]) if data.get("gsd") else None,
    )
    job_id = uuid.uuid4().hex
    job_dir = OUT_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / "mosaico.tif"
    q: queue.Queue = queue.Queue()
    JOBS[job_id] = {"q": q, "out": out_path,
                    "preview": out_path.with_suffix(".preview.png"), "report": None}

    def run():
        def cb(phase, msg, i=None, n=None):
            q.put({"phase": phase, "msg": msg, "i": i, "n": n})
        try:
            report = build_mosaic([folder], out_path, opts, cb)
            JOBS[job_id]["report"] = report
            q.put({"phase": "done", "report": report})
        except Exception as e:  # noqa: BLE001
            q.put({"phase": "error", "msg": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/events/<job_id>")
def events(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="job desconocido"), 404

    def gen():
        while True:
            ev = job["q"].get()
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev["phase"] in ("done", "error"):
                break

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/preview/<job_id>")
def preview(job_id):
    job = JOBS.get(job_id)
    if not job or not Path(job["preview"]).exists():
        return jsonify(error="sin vista previa"), 404
    return send_file(job["preview"], mimetype="image/png")


@app.get("/api/result/<job_id>")
def result(job_id):
    job = JOBS.get(job_id)
    if not job or not Path(job["out"]).exists():
        return jsonify(error="sin resultado"), 404
    return send_file(job["out"], mimetype="image/tiff",
                     as_attachment=True, download_name="mosaico.tif")


if __name__ == "__main__":
    # Server de desarrollo (local). En producción corre bajo gunicorn (ver Dockerfile),
    # que importa `app` directamente y no pasa por este bloque.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("MOSAICO_HOST_ALL") == "1" else "127.0.0.1"
    print(f"Mosaico-Drone — http://127.0.0.1:{port}")
    app.run(host=host, port=port, threaded=True)
