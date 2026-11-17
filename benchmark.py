"""
Paso 6b: benchmark de escalabilidad.

Compara busqueda exacta (IndexFlatIP) contra aproximada (IndexHNSWFlat)
a distintos tamanos de indice, midiendo latencia y perdida de calidad.

Uso:
    python benchmark.py

Produce:
    resultados/benchmark.csv
    resultados/benchmark.png
"""

import csv
import os
import time

import faiss
import numpy as np

K = 6
N_CONSULTAS = 300
TAMANOS = [1000, 5000, 10000, 25250]
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64
OUT = "resultados"
os.makedirs(OUT, exist_ok=True)

print("Cargando embeddings...")
embs = np.load("data/index/embeddings.npy")
N, DIM = embs.shape
print(f"{N} vectores de {DIM} dimensiones\n")

rng = np.random.default_rng(42)
consultas = embs[rng.choice(N, N_CONSULTAS, replace=False)].copy()


def medir(index, consultas, k):
    """Devuelve (p50_ms, p95_ms, ids) midiendo consulta por consulta."""
    tiempos = []
    todos = []
    for q in consultas:
        q = q.reshape(1, -1)
        t0 = time.perf_counter()
        _, ids = index.search(q, k)
        tiempos.append((time.perf_counter() - t0) * 1000)
        todos.append(ids[0])
    t = np.array(tiempos)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95)), np.array(todos)


filas = []
for n in TAMANOS:
    if n > N:
        continue
    sub = np.ascontiguousarray(embs[:n])
    print(f"--- {n} vectores ---")

    flat = faiss.IndexFlatIP(DIM)
    flat.add(sub)
    p50_f, p95_f, ids_f = medir(flat, consultas, K)
    print(f"  Flat  p50={p50_f:6.3f} ms  p95={p95_f:6.3f} ms")

    t0 = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw.add(sub)
    seg_build = time.perf_counter() - t0
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH
    p50_h, p95_h, ids_h = medir(hnsw, consultas, K)

    # Recall@K: fraccion del top-K exacto que HNSW tambien recupera
    coincidencias = [len(set(a) & set(b)) for a, b in zip(ids_f, ids_h)]
    recall = float(np.mean(coincidencias) / K)

    print(f"  HNSW  p50={p50_h:6.3f} ms  p95={p95_h:6.3f} ms  "
          f"recall@{K}={recall:.4f}  (construccion {seg_build:.1f} s)")

    filas.append({
        "n_vectores": n,
        "flat_p50_ms": round(p50_f, 4),
        "flat_p95_ms": round(p95_f, 4),
        "hnsw_p50_ms": round(p50_h, 4),
        "hnsw_p95_ms": round(p95_h, 4),
        "hnsw_recall_at_6": round(recall, 4),
        "hnsw_build_s": round(seg_build, 2),
        "aceleracion": round(p50_f / p50_h, 2) if p50_h > 0 else None,
    })

ruta_csv = os.path.join(OUT, "benchmark.csv")
with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
    w.writeheader()
    w.writerows(filas)
print(f"\nCSV: {ruta_csv}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [f["n_vectores"] for f in filas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ns, [f["flat_p50_ms"] for f in filas], "o-",
             color="#378ADD", label="Flat (exacto) p50")
    ax1.plot(ns, [f["flat_p95_ms"] for f in filas], "o--",
             color="#378ADD", alpha=0.45, label="Flat p95")
    ax1.plot(ns, [f["hnsw_p50_ms"] for f in filas], "s-",
             color="#D85A30", label="HNSW (aprox.) p50")
    ax1.plot(ns, [f["hnsw_p95_ms"] for f in filas], "s--",
             color="#D85A30", alpha=0.45, label="HNSW p95")
    ax1.set_xlabel("Vectores en el indice")
    ax1.set_ylabel("Latencia por consulta (ms)")
    ax1.set_title("Latencia vs tamano del indice")
    ax1.legend()
    ax1.grid(alpha=0.25)

    acc = [f["aceleracion"] for f in filas]
    ax2.plot(ns, acc, "o-", color="#1D9E75", lw=2)
    ax2.axhline(1.0, color="#888780", ls=":", lw=1)
    for x, y, f in zip(ns, acc, filas):
        ax2.annotate(f"{y:.1f}x\nrecall {f['hnsw_recall_at_6']:.4f}", (x, y),
                     textcoords="offset points", xytext=(-18, 10),
                     fontsize=8.5, color="#0F6E56")
    ax2.set_xlabel("Vectores en el indice")
    ax2.set_ylabel("Aceleracion de HNSW sobre Flat")
    ax2.set_title("Ventaja de la busqueda aproximada")
    ax2.set_ylim(0, max(acc) * 1.3)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Escalabilidad: {N_CONSULTAS} consultas, K={K}", fontsize=13)
    fig.tight_layout()
    ruta_png = os.path.join(OUT, "benchmark.png")
    fig.savefig(ruta_png, dpi=140)
    print(f"Grafico: {ruta_png}")
except ImportError:
    print("(matplotlib no instalado: se omite el grafico)")
