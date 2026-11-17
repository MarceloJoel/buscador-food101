"""
Paso 6a: evaluacion cuantitativa del buscador.

Para cada una de las 101 clases lanza la consulta "a photo of <clase>"
y mide que fraccion del top-6 pertenece realmente a esa clase.

    Precision@6 = aciertos / 6, promediado sobre las 101 clases

Uso:
    python evaluar.py

Produce:
    resultados/precision_por_clase.csv
    resultados/precision_por_clase.png
"""

import csv
import os
import time

import numpy as np

from api import motor

K = 6
PLANTILLA = "a photo of {}"
OUT = "resultados"
os.makedirs(OUT, exist_ok=True)

print("Cargando modelo e indice...")
motor.cargar()
clases = motor._E["clases"]
items = motor._E["items"]
index = motor._E["index"]
print(f"{index.ntotal} vectores, {len(clases)} clases\n")

etiquetas_idx = np.array([it["label_id"] for it in items], dtype="int32")

filas = []
t0 = time.perf_counter()

for cid, nombre in enumerate(clases):
    consulta = PLANTILLA.format(nombre.replace("_", " "))
    v = motor.embeber_texto(consulta)
    _, ids = index.search(v, K)
    recuperadas = etiquetas_idx[ids[0]]
    aciertos = int((recuperadas == cid).sum())
    confusas = [clases[int(e)] for e in recuperadas if int(e) != cid]
    filas.append({
        "clase": nombre,
        "consulta": consulta,
        "aciertos": aciertos,
        "precision_at_6": round(aciertos / K, 4),
        "confundida_con": "; ".join(confusas),
    })

segundos = time.perf_counter() - t0

precisiones = np.array([f["precision_at_6"] for f in filas])
media = precisiones.mean()
perfectas = int((precisiones == 1.0).sum())
ceros = int((precisiones == 0.0).sum())

print("=" * 58)
print(f"  Precision@{K} media      : {media:.4f}  ({media*100:.1f}%)")
print(f"  Clases con 6/6 aciertos : {perfectas} de {len(clases)}")
print(f"  Clases con 0/6 aciertos : {ceros} de {len(clases)}")
print(f"  Mediana                 : {np.median(precisiones):.4f}")
print(f"  Tiempo total            : {segundos:.1f} s "
      f"({segundos/len(clases)*1000:.1f} ms por consulta)")
print("=" * 58)

orden = sorted(filas, key=lambda f: -f["precision_at_6"])
print("\nMejores 10:")
for f in orden[:10]:
    print(f"  {f['precision_at_6']:.2f}  {f['clase']}")
print("\nPeores 10:")
for f in orden[-10:]:
    print(f"  {f['precision_at_6']:.2f}  {f['clase']}")

print("\nConfusiones (clases que no llegaron a 6/6):")
for f in sorted((x for x in filas if x["aciertos"] < K),
                key=lambda x: x["precision_at_6"]):
    print(f"  {f['clase']:20s} {f['aciertos']}/{K}  ->  {f['confundida_con']}")

ruta_csv = os.path.join(OUT, "precision_por_clase.csv")
with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["clase", "consulta", "aciertos", "precision_at_6", "confundida_con"])
    w.writeheader()
    w.writerows(filas)
print(f"\nCSV: {ruta_csv}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    valores, cuentas = np.unique(precisiones, return_counts=True)
    ax1.bar([f"{v:.2f}" for v in valores], cuentas, color="#1D9E75")
    ax1.set_title(f"Distribucion de Precision@{K} (101 clases)")
    ax1.set_xlabel(f"Precision@{K}")
    ax1.set_ylabel("Numero de clases")

    imperf = sorted((f for f in filas if f["aciertos"] < K),
                    key=lambda f: f["precision_at_6"])
    ax2.barh([f["clase"].replace("_", " ") for f in imperf],
             [f["precision_at_6"] for f in imperf], color="#D85A30", height=0.6)
    for i, f in enumerate(imperf):
        ax2.text(f["precision_at_6"] + 0.015, i,
                 f["confundida_con"].replace("; ", ", ").replace("_", " "),
                 va="center", fontsize=8.5, color="#5F5E5A")
    ax2.set_title(f"Las {len(imperf)} clases imperfectas y sus confusiones")
    ax2.set_xlabel(f"Precision@{K}")
    ax2.set_xlim(0, 1.35)

    fig.suptitle(f"Precision@{K} media = {media:.3f}", fontsize=13)
    fig.tight_layout()
    ruta_png = os.path.join(OUT, "precision_por_clase.png")
    fig.savefig(ruta_png, dpi=140)
    print(f"Grafico: {ruta_png}")
except ImportError:
    print("(matplotlib no instalado: se omite el grafico)")
