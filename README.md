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
| 25,250 | 3.693 ms | 0.195 ms | 19.0× | 0.9989 |

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
├── ui_comun.py             Estilos, portada, grilla de resultados y estado del backend
├── static/
│   └── portada.jpg         Fondo del encabezado de ambas vistas
├── .streamlit/
│   └── config.toml         Tema claro y servido de static/ (enableStaticServing)
├── construir_indice.py     Genera embeddings, miniaturas e índice FAISS
├── evaluar.py              Precisión@6 sobre las 101 clases
├── benchmark.py            Latencia y recall: Flat vs HNSW
├── poc_busqueda.py         Prueba de concepto con 50 imágenes
├── data/                   Generado (no versionado)
│   ├── images/             25,250 miniaturas (lado máximo 320 px)
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

> **Importante:** todos los comandos de esta sección (`python`, `uvicorn`, `streamlit`) están instalados **dentro del entorno virtual**. Cada terminal nueva debe activarlo primero desde la raíz del proyecto; si no, PowerShell responde `El término 'uvicorn' no se reconoce...`.
>
> ```powershell
> cd buscador-food101
> .\.venv\Scripts\Activate.ps1
> ```
>
> El prompt debe mostrar `(.venv)` al inicio. Si PowerShell bloquea el script por la política de ejecución, habilítala una vez para tu usuario:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> Alternativa sin activar: invocar los ejecutables del entorno directamente, por ejemplo `.\.venv\Scripts\uvicorn.exe api.main:app --port 8000`.

### 1. Construir el índice

Con el entorno activado:

```powershell
python construir_indice.py
```

Descarga Food-101 desde Hugging Face (~5 GB, una sola vez), calcula los embeddings de las 25,250 imágenes de validación y escribe `data/index/`. Toma unos 10 minutos en GPU.

### 2. Levantar la API (terminal 1)

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --port 8000
```

Para desarrollo, con recarga automática al guardar cambios en el backend:

```powershell
uvicorn api.main:app --reload --reload-dir api --port 8000
```

`--reload-dir api` evita que vigile `data/` (25,250 imágenes) y `.venv/`.

Documentación interactiva en `http://localhost:8000/docs`. El frontend espera la API en `http://localhost:8000`, así que conserva ese puerto.

### 3. Levantar el frontend (terminal 2)

En otra terminal, también con el entorno activado:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Disponible en `http://localhost:8501`. Tiene dos páginas:

| Página | Qué hace |
|--------|----------|
| **Buscar por Descripción** | Escribe un plato (mejor en inglés) o elige uno de los ejemplos. También acepta la consulta en la URL: `http://localhost:8501/?q=a plate of sushi` |
| **Buscar por Foto** | Sube una foto (JPG, PNG, WebP o BMP, hasta 10 MB) y pulsa *Buscar platos parecidos* |

Ambas muestran los 6 resultados con su nombre y su similitud. La barra lateral indica si la API está encendida y cuántas imágenes tiene el índice.

El encabezado de ambas vistas (título, buscador, ejemplos y nota en *Buscar por Descripción*; título y zona de subida en *Buscar por Foto*) se muestra sobre `static/portada.jpg`. Streamlit sirve esa carpeta en `app/static/` gracias a `enableStaticServing = true` en `.streamlit/config.toml`.

> Streamlit lee `.streamlit/config.toml` solo al arrancar. Si cambias ese archivo, o si la portada aparece sin foto, detén el frontend (`Ctrl+C`) y vuelve a ejecutar `streamlit run app.py`. Los cambios en archivos `.py` se recargan solos (`runOnSave = true`).

### 4. Reproducir la evaluación

Con el entorno activado:

```powershell
python evaluar.py
python benchmark.py
```

Requieren el índice de `data/index/`; no necesitan la API encendida.

| Script | Qué mide | Salida en `resultados/` |
|--------|----------|-------------------------|
| `evaluar.py` | Precisión@6 con una consulta por clase (`"a photo of {clase}"`) | `precision_por_clase.csv`, `precision_por_clase.png` |
| `benchmark.py` | Latencia p50/p95 y Recall@6 de Flat vs HNSW con 300 consultas (HNSW: M=32, efConstruction=80, efSearch=64) | `benchmark.csv`, `benchmark.png` |

---

## API

| Método | Ruta | Uso |
|--------|------|-----|
| `GET` | `/health` | Estado del modelo y del índice |
| `POST` | `/search/text` | Texto → imagen |
| `POST` | `/search/image` | Imagen → imagen |
| `GET` | `/images/{archivo}` | Miniaturas del índice (las `url` de cada resultado) |
| `GET` | `/docs` | Documentación interactiva |

### `POST /search/text`

```json
{ "query": "a plate of sushi", "top_k": 6 }
```

`query` admite de 1 a 300 caracteres y `top_k` va de 1 a 50 (por defecto 6).

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

`multipart/form-data` con los campos `file` (máximo 10 MB) y `top_k` (1 a 50, por defecto 6). Devuelve la misma estructura. La API acepta cualquier formato que Pillow pueda abrir; el frontend limita la subida a JPG, PNG, WebP y BMP.

### `GET /health`

Estado del modelo y del índice, sin ejecutar una búsqueda:

```json
{ "listo": true, "dispositivo": "cuda", "modelo": "ViT-B-32-quickgelu",
  "pesos": "openai", "dim": 512, "vectores": 25250, "clases": 101 }
```

### Errores

| Código | Causa |
|--------|-------|
| `503` | El índice no está cargado: ejecuta `construir_indice.py` y reinicia la API |
| `413` | La imagen supera los 10 MB |
| `422` | La imagen no se puede leer, o `query` / `top_k` están fuera de rango |

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
