"""
main.py — API FastAPI del buscador multimodal.

Arrancar desde la raiz del proyecto:
    uvicorn api.main:app --port 8000

No usar --reload: el vigilante rastrea data/images (25 250 archivos) y .venv,
y reinicia la API cada vez que se guarda cualquier .py, incluidos los de
Streamlit. Para desarrollar sobre la API: --reload --reload-dir api

Endpoints:
    GET  /health         estado del modelo y del indice
    POST /search/text    texto -> imagen
    POST /search/image   imagen -> imagen
    GET  /images/{nombre} miniaturas estaticas
    GET  /docs           documentacion interactiva
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import motor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("buscador")

MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Cargando modelo CLIP e indice FAISS...")
    try:
        motor.cargar()
        log.info("Listo: %s", motor.estado())
    except Exception as e:
        log.warning("No se pudo cargar el indice: %s", e)
    yield
    log.info("Apagando.")


app = FastAPI(
    title="Buscador multimodal de imagenes - Food-101",
    description="Busqueda texto->imagen e imagen->imagen con CLIP y FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=motor.DIR_IMGS), name="images")


class ConsultaTexto(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, examples=["a plate of sushi"])
    top_k: int = Field(6, ge=1, le=50)


@app.get("/health")
def health():
    return motor.estado()


@app.post("/search/text")
def search_text(body: ConsultaTexto):
    if not motor._E.get("listo"):
        raise HTTPException(503, "Indice no cargado. Ejecuta construir_indice.py")
    return motor.buscar_texto(body.query, body.top_k)


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = Form(6)):
    if not motor._E.get("listo"):
        raise HTTPException(503, "Indice no cargado. Ejecuta construir_indice.py")
    if not 1 <= top_k <= 50:
        raise HTTPException(422, "top_k debe estar entre 1 y 50")

    datos = await file.read()
    if len(datos) > MAX_BYTES:
        raise HTTPException(413, "La imagen supera los 10 MB")
    try:
        img = Image.open(io.BytesIO(datos))
        img.load()
    except Exception:
        raise HTTPException(422, "No se pudo leer la imagen (usa JPG, PNG o WebP)")

    return motor.buscar_imagen(img, top_k)
