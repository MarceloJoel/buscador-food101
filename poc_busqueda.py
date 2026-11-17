"""
Prueba de concepto: busqueda texto -> imagen con 50 fotos de Food-101.

Uso:
    python poc_busqueda.py

Salida:
    - TOP 6 en consola
    - resultado.jpg con la grilla de 6 imagenes
"""

import numpy as np
import torch
import open_clip
from datasets import load_dataset
from PIL import Image

# ----------------------------------------------------------------------
# Parametros: cambia CONSULTA por lo que quieras buscar
# ----------------------------------------------------------------------
N = 50
CONSULTA = "a plate of sushi"

# ----------------------------------------------------------------------
# 1) Modelo CLIP
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

print("Cargando CLIP (la primera vez descarga ~350 MB)...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

# ----------------------------------------------------------------------
# 2) Dataset
# ----------------------------------------------------------------------
print("Cargando dataset (la primera vez descarga ~1.3 GB)...")
ds = load_dataset("ethz/food101", split="validation")
ds = ds.shuffle(seed=42).select(range(N))
nombres = ds.features["label"].names

# ----------------------------------------------------------------------
# 3) Embeddings de las imagenes
# ----------------------------------------------------------------------
print(f"Calculando embeddings de {N} imagenes...")
imgs = [ds[i]["image"].convert("RGB") for i in range(N)]
lote = torch.stack([preprocess(im) for im in imgs]).to(device)

with torch.no_grad():
    emb = model.encode_image(lote)
    emb = emb / emb.norm(dim=-1, keepdim=True)

emb = emb.cpu().numpy().astype("float32")
print("Matriz de embeddings:", emb.shape)

# ----------------------------------------------------------------------
# 4) Embedding de la consulta de texto
# ----------------------------------------------------------------------
print(f"Buscando: '{CONSULTA}'")
tok = tokenizer([CONSULTA]).to(device)

with torch.no_grad():
    q = model.encode_text(tok)
    q = q / q.norm(dim=-1, keepdim=True)

q = q.cpu().numpy().astype("float32")[0]

# ----------------------------------------------------------------------
# 5) Similitud coseno = producto punto (vectores normalizados)
# ----------------------------------------------------------------------
sims = emb @ q
top = np.argsort(-sims)[:6]

print("\n--- TOP 6 ---")
for r, i in enumerate(top, 1):
    etiqueta = nombres[ds[int(i)]["label"]]
    print(f"{r}. {etiqueta:25s} similitud = {sims[i]:.4f}")

# ----------------------------------------------------------------------
# 6) Hoja de contacto con los 6 resultados
# ----------------------------------------------------------------------
W = 224
hoja = Image.new("RGB", (W * 3, W * 2), "white")
for pos, i in enumerate(top):
    hoja.paste(imgs[int(i)].resize((W, W)), ((pos % 3) * W, (pos // 3) * W))
hoja.save("resultado.jpg", quality=90)

print("\nGuardado: resultado.jpg  <- abrelo")
