# Buscador multimodal de imágenes — Food-101

Sistema de búsqueda de imágenes por **consultas textuales** y **por similaridad de imágenes**, construido sobre **CLIP** para la extracción de embeddings y **FAISS** para la búsqueda vectorial.

Proyecto final del curso *Tópicos Avanzados en Machine Learning* — 2026-II.

---

## Qué hace

| Modo | Entrada | Salida |
|------|---------|--------|
| Texto → imagen | `"a plate of sushi"` | Las 6 imágenes más relevantes del índice |
| Imagen → imagen | Una foto subida por el usuario | Las 6 imágenes visualmente más parecidas |

CLIP proyecta texto e imágenes al **mismo espacio vectorial de 512 dimensiones**, de modo que ambas modalidades de consulta operan sobre un único índice. La similitud es el coseno, que con vectores normalizados L2 equivale al producto interno.

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Imágenes indexadas | 25,250 (split de validación de Food-101) |
| Clases | 101 |
| Tiempo de indexado | 9.3 min en GPU NVIDIA GTX 1650 (45.2 img/s) |
| **Precisión@6** | **0.9752** sobre 101 consultas |
| Clases con 6/6 aciertos | 93 de 101 |
| Latencia de búsqueda (exacta) | 3.69 ms (p50) |
| Latencia de búsqueda (HNSW) | 0.195 ms (p50) |

### Escalabilidad

| Vectores | Flat (exacto) | HNSW (aprox.) | Aceleración | Recall@6 |
|---------:|--------------:|--------------:|------------:|---------:|
| 1,000 | 0.052 ms | 0.069 ms | 0.75× | 1.0000 |
| 5,000 | 0.274 ms | 0.106 ms | 2.6× | 0.9989 |
| 10,000 | 0.828 ms | 0.172 ms | 4.8× | 0.9983 |
| 25,250 | 3.693 ms | 0.195 ms | 18.9× | 0.9989 |

La búsqueda exacta crece linealmente con el número de vectores; HNSW crece de forma logarítmica. El punto de cruce está en torno a los 2,000–3,000 vectores: por debajo de esa cifra, el índice aproximado no compensa su propio sobrecosto.

---

## Arquitectura

```
Indexación (una sola vez, offline)
  Food-101 ──> CLIP image encoder ──> vectores 512-d ──> FAISS + metadatos

Consulta (en línea, milisegundos)
  texto  ──> CLIP text encoder  ──┐
  imagen ──> CLIP image encoder ──┴──> vector ──> FAISS top-6 ──> JSON
```

El diseño separa **indexación** de **consulta**: el índice se reconstruye sin tocar el servicio que atiende peticiones.

```
buscador-food101/
├── api/
│   ├── motor.py            Carga de CLIP y FAISS; funciones de búsqueda
│   └── main.py             API FastAPI
├── vistas/
│   ├── texto.py            Interfaz A: texto → imagen
│   └── imagen.py           Interfaz B: imagen → imagen
├── app.py                  Navegación del frontend (Streamlit)
├── ui_comun.py             Grilla de resultados y estado del backend
├── construir_indice.py     Genera embeddings, miniaturas e índice FAISS
├── evaluar.py              Precisión@6 sobre las 101 clases
├── benchmark.py            Latencia y recall: Flat vs HNSW
├── poc_busqueda.py         Prueba de concepto con 50 imágenes
├── data/                   Generado (no versionado)
│   ├── images/             25,250 miniaturas
│   └── index/              embeddings.npy, flat.faiss, meta.json
└── resultados/             CSV y gráficos de la evaluación
```

---

## Instalación

Requiere **Python 3.12**. PyTorch aún no publica wheels de CUDA para 3.14, por lo que esa versión dejaría la GPU sin usar.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Verifica que la GPU esté disponible:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Uso

### 1. Construir el índice

```powershell
python construir_indice.py
```

Descarga Food-101 desde Hugging Face (~5 GB, una sola vez), calcula los embeddings de las 25,250 imágenes de validación y escribe `data/index/`. Toma unos 10 minutos en GPU.

### 2. Levantar la API

```powershell
uvicorn api.main:app --port 8000
```

Documentación interactiva en `http://localhost:8000/docs`.

### 3. Levantar el frontend

En otra terminal, con el entorno activado:

```powershell
streamlit run app.py
```

Disponible en `http://localhost:8501`.

### 4. Reproducir la evaluación

```powershell
python evaluar.py
python benchmark.py
```

Los resultados quedan en `resultados/`.

---

## API

### `POST /search/text`

```json
{ "query": "a plate of sushi", "top_k": 6 }
```

Respuesta:

```json
{
  "resultados": [
    { "id": 445, "archivo": "00445.jpg", "label": "sushi",
      "url": "/images/00445.jpg", "score": 0.3343 }
  ],
  "n": 6,
  "ms": 8.01
}
```

### `POST /search/image`

`multipart/form-data` con los campos `file` (JPG, PNG, WebP o BMP, máximo 10 MB) y `top_k`. Devuelve la misma estructura.

### `GET /health`

Estado del modelo y del índice, sin ejecutar una búsqueda.

---

## Modelo

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | `ViT-B-32-quickgelu` |
| Pesos | `openai` |
| Dimensión | 512 |
| Librería | `open-clip-torch` |

El sufijo `quickgelu` es necesario: los pesos originales de OpenAI fueron entrenados con la activación QuickGELU, y cargarlos en la configuración `ViT-B-32` estándar produce un desajuste que degrada la precisión.

---

## Dataset

[`ethz/food101`](https://huggingface.co/datasets/ethz/food101) — 101,000 imágenes en 101 categorías de comida, publicado por ETH Zurich. Este proyecto indexa el split de validación (25,250 imágenes, 250 por clase).

Las etiquetas **no se usan para buscar**: sirven únicamente como referencia para medir la Precisión@6.
