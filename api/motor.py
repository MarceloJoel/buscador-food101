"""
motor.py — carga del modelo CLIP y del indice FAISS, y funciones de busqueda.

Se carga una sola vez (patron singleton) y lo reutilizan la API y los
scripts de evaluacion.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
import open_clip
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(BASE, "data")
DIR_INDEX = os.path.join(DIR_DATA, "index")
DIR_IMGS = os.path.join(DIR_DATA, "images")

_lock = threading.Lock()
_E: Dict[str, Any] = {"listo": False}


def cargar() -> None:
    """Carga modelo, tokenizer, indice y metadatos. Idempotente."""
    if _E.get("listo"):
        return
    with _lock:
        if _E.get("listo"):
            return

        ruta_meta = os.path.join(DIR_INDEX, "meta.json")
        if not os.path.exists(ruta_meta):
            raise FileNotFoundError(
                "Falta data/index/meta.json. Ejecuta primero construir_indice.py"
            )

        with open(ruta_meta, encoding="utf-8") as f:
            meta = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            meta["modelo"], pretrained=meta["pesos"]
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(meta["modelo"])

        index = faiss.read_index(os.path.join(DIR_INDEX, "flat.faiss"))

        _E.update({
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "index": index,
            "meta": meta,
            "items": meta["items"],
            "clases": meta["clases"],
            "device": device,
            "listo": True,
        })


def estado() -> Dict[str, Any]:
    if not _E.get("listo"):
        return {"listo": False}
    return {
        "listo": True,
        "dispositivo": _E["device"],
        "modelo": _E["meta"]["modelo"],
        "pesos": _E["meta"]["pesos"],
        "dim": _E["meta"]["dim"],
        "vectores": int(_E["index"].ntotal),
        "clases": len(_E["clases"]),
    }


def _formatear(scores, ids) -> List[Dict[str, Any]]:
    items = _E["items"]
    salida = []
    for s, i in zip(scores, ids):
        if i < 0:
            continue
        it = items[int(i)]
        salida.append({
            "id": it["id"],
            "archivo": it["archivo"],
            "label": it["label"],
            "url": f"/images/{it['archivo']}",
            "score": round(float(s), 4),
        })
    return salida


def _buscar(vector: np.ndarray, k: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    scores, ids = _E["index"].search(vector, k)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "resultados": _formatear(scores[0], ids[0]),
        "n": int(len(ids[0])),
        "ms": round(ms, 2),
    }


def embeber_texto(consulta: str) -> np.ndarray:
    cargar()
    tok = _E["tokenizer"]([consulta]).to(_E["device"])
    with torch.no_grad():
        v = _E["model"].encode_text(tok)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype("float32")


def embeber_imagen(img: Image.Image) -> np.ndarray:
    cargar()
    lote = _E["preprocess"](img.convert("RGB")).unsqueeze(0).to(_E["device"])
    with torch.no_grad():
        v = _E["model"].encode_image(lote)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype("float32")


def buscar_texto(consulta: str, k: int = 6) -> Dict[str, Any]:
    """Busqueda multimodal texto -> imagen."""
    return _buscar(embeber_texto(consulta), k)


def buscar_imagen(img: Image.Image, k: int = 6) -> Dict[str, Any]:
    """Busqueda por similaridad imagen -> imagen."""
    return _buscar(embeber_imagen(img), k)
