"""
Paso 3: indexa Food-101 con CLIP y construye el indice FAISS.

Uso:
    python construir_indice.py

Produce:
    data/images/00000.jpg ...      miniaturas para el frontend
    data/index/embeddings.npy      matriz (N, 512) float32 normalizada L2
    data/index/meta.json           id -> archivo, clase
    data/index/flat.faiss          indice FAISS exacto (producto interno)
"""

import json
import os
import time

import faiss
import numpy as np
import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------------------------------------------------
# Parametros
# ----------------------------------------------------------------------
SPLIT = "validation"       # 25.250 imagenes. Cambia a "train" para 75.750
BATCH = 64                 # bajalo a 32 si la GPU se queda sin memoria
THUMB = 320                # lado maximo de las miniaturas, en pixeles
OUT = "data"
MODELO = "ViT-B-32-quickgelu"
PESOS = "openai"

os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/index", exist_ok=True)

# ----------------------------------------------------------------------
# 1) Modelo
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(MODELO, pretrained=PESOS)
model = model.to(device).eval()
DIM = model.visual.output_dim
print(f"Modelo: {MODELO}/{PESOS}  |  dimension = {DIM}")

# ----------------------------------------------------------------------
# 2) Dataset
# ----------------------------------------------------------------------
print(f"Cargando split '{SPLIT}' (ya esta en cache)...")
ds = load_dataset("ethz/food101", split=SPLIT).shuffle(seed=42)
clases = ds.features["label"].names
N = len(ds)
print(f"{N} imagenes, {len(clases)} clases")

# ----------------------------------------------------------------------
# 3) Embeddings por lotes + miniaturas
# ----------------------------------------------------------------------
embs = np.zeros((N, DIM), dtype="float32")
meta = []

t0 = time.time()
for ini in tqdm(range(0, N, BATCH), desc="Indexando", unit="lote"):
    fin = min(ini + BATCH, N)
    filas = ds[ini:fin]

    imgs = [im.convert("RGB") for im in filas["image"]]
    lote = torch.stack([preprocess(im) for im in imgs]).to(device)

    with torch.no_grad():
        e = model.encode_image(lote)
        e = e / e.norm(dim=-1, keepdim=True)
    embs[ini:fin] = e.cpu().numpy().astype("float32")

    for k, im in enumerate(imgs):
        i = ini + k
        archivo = f"{i:05d}.jpg"
        th = im.copy()
        th.thumbnail((THUMB, THUMB))
        th.save(os.path.join(OUT, "images", archivo), quality=85)
        etiqueta = int(filas["label"][k])
        meta.append({
            "id": i,
            "archivo": archivo,
            "label_id": etiqueta,
            "label": clases[etiqueta],
        })

segundos = time.time() - t0
print(f"\nEmbeddings listos en {segundos/60:.1f} min "
      f"({N/segundos:.1f} imagenes/s)")

# ----------------------------------------------------------------------
# 4) Guardar artefactos
# ----------------------------------------------------------------------
np.save(f"{OUT}/index/embeddings.npy", embs)

with open(f"{OUT}/index/meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "modelo": MODELO,
        "pesos": PESOS,
        "dim": DIM,
        "split": SPLIT,
        "n": N,
        "segundos_indexado": round(segundos, 1),
        "clases": clases,
        "items": meta,
    }, f, ensure_ascii=False)

# ----------------------------------------------------------------------
# 5) Indice FAISS exacto
# ----------------------------------------------------------------------
# Con vectores normalizados L2, el producto interno equivale al coseno.
index = faiss.IndexFlatIP(DIM)
index.add(embs)
faiss.write_index(index, f"{OUT}/index/flat.faiss")
print(f"Indice FAISS: {index.ntotal} vectores de {DIM} dimensiones")

# ----------------------------------------------------------------------
# 6) Prueba rapida
# ----------------------------------------------------------------------
tokenizer = open_clip.get_tokenizer(MODELO)
tok = tokenizer(["a plate of sushi"]).to(device)
with torch.no_grad():
    q = model.encode_text(tok)
    q = q / q.norm(dim=-1, keepdim=True)
q = q.cpu().numpy().astype("float32")

t0 = time.time()
sims, ids = index.search(q, 6)
ms = (time.time() - t0) * 1000

print(f"\nPrueba: 'a plate of sushi'  ({ms:.2f} ms)")
for r, (s, i) in enumerate(zip(sims[0], ids[0]), 1):
    print(f"  {r}. {meta[i]['label']:25s} {s:.4f}  {meta[i]['archivo']}")

print("\nListo. Artefactos en data/index/")
