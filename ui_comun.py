"""Helpers compartidos por las dos vistas de Streamlit (estilos, estado, resultados)."""

from html import escape

import requests
import streamlit as st

API = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Estilos: "La mesa" — los resultados se sirven como platos vistos desde arriba.
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Young+Serif&display=swap');

:root {
  --arroz:   #F6F7F2;
  --albahaca:#17332A;
  --azafran: #F0A202;
  --salvia:  #6E8577;
  --hoja:    #E3E9DF;
  --plato:   #FFFFFF;
  --serif: 'Young Serif', Georgia, serif;
  --sans:  'Figtree', system-ui, sans-serif;
}

.stApp, .stApp p, .stApp li, .stApp label, .stApp input,
.stApp button, .stApp textarea, .stApp h1, .stApp h2, .stApp h3 {
  font-family: var(--sans);
}
.stApp { background: var(--arroz); color: var(--albahaca); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1080px; padding-top: 4.5rem; }

/* ---------- Encabezado ---------- */
.hero { margin: 0 0 2rem; max-width: 40rem; }
.hero [data-testid="stHeaderActionElements"],
.mesa-cab [data-testid="stHeaderActionElements"] { display: none; }
.hero h1, .hero h1 span, .mesa-cab h2, .mesa-cab h2 span { font-family: var(--serif) !important; }
.hero h1 {
  font-family: var(--serif);
  font-weight: 400;
  font-size: clamp(2.6rem, 6vw, 4.4rem);
  line-height: 1.02;
  letter-spacing: -0.02em;
  color: var(--albahaca);
  margin: 0 0 1rem;
  padding: 0;
}
.hero p {
  font-size: 1.125rem;
  line-height: 1.55;
  color: var(--salvia);
  margin: 0;
}

/* ---------- Campo de texto ---------- */
[data-testid="stTextInputRootElement"] {
  background: var(--plato);
  border: 1.5px solid #D3DBCF;
  border-radius: 999px;
  min-height: 3.4rem;
  padding: 0 0.9rem;
  transition: border-color .15s ease;
}
[data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--albahaca);
  box-shadow: 0 0 0 3px rgba(240,162,2,.45);
}
[data-testid="stTextInputField"] {
  font-size: 1.125rem;
  color: var(--albahaca);
  background: transparent;
}
[data-testid="stTextInputField"]::placeholder { color: #9AAA9F; }

/* ---------- Botones ---------- */
.stButton button, .stFormSubmitButton button {
  border-radius: 999px;
  font-weight: 600;
  min-height: 3.4rem;
  padding: 0 1.6rem;
}
[data-testid="stBaseButton-primary"] {
  background: var(--albahaca);
  border: 0;
  color: var(--arroz);
}
[data-testid="stBaseButton-primary"]:hover {
  background: #0F241D;
  color: #fff;
}
button:focus-visible,
[data-testid="stBaseButton-pills"]:focus-visible,
[data-testid="stBaseButton-pillsActive"]:focus-visible {
  outline: 3px solid var(--azafran) !important;
  outline-offset: 2px;
  box-shadow: none !important;
}

/* ---------- Chips de ejemplo ---------- */
[data-testid="stPills"] { margin-top: -.25rem; }
[data-testid="stBaseButton-pills"],
[data-testid="stBaseButton-pillsActive"] {
  border-radius: 999px;
  border: 1px solid var(--hoja);
  background: transparent;
  color: var(--albahaca);
  font-size: .95rem;
  padding: .3rem .9rem;
}
[data-testid="stBaseButton-pills"]:hover { background: var(--hoja); border-color: var(--hoja); }
[data-testid="stBaseButton-pillsActive"] {
  background: var(--albahaca); color: var(--arroz); border-color: var(--albahaca);
}
.nota { color: var(--salvia); font-size: .9rem; margin: .35rem 0 0; }

/* ---------- Subida de imagen ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--plato);
  border: 2px dashed #C9D3C5;
  border-radius: 28px;
  padding: 2.2rem 1.6rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--albahaca); }
[data-testid="stFileUploaderDropzone"] button { border-radius: 999px; }

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"] { background: var(--albahaca); }
[data-testid="stSidebar"] * { color: #DCE5DE; }
[data-testid="stSidebarNav"] a { border-radius: 10px; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(246,247,242,.12); }
[data-testid="stSidebarNav"] a p { color: var(--arroz) !important; font-weight: 500; }
[data-testid="stSidebarNav"] [data-testid="stIconMaterial"] { color: var(--azafran) !important; }
.estado { margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid rgba(220,229,222,.18); }
.estado h3 {
  font-family: var(--serif); font-weight: 400; font-size: 1.15rem;
  color: var(--arroz) !important; margin: 0 0 .9rem; padding: 0;
}
.estado .luz { display: inline-flex; align-items: center; gap: .5rem; font-weight: 600; }
.estado .luz::before {
  content: ""; width: .6rem; height: .6rem; border-radius: 50%; background: var(--c);
}
.estado .cifra {
  font-family: var(--serif); font-size: 2.2rem; line-height: 1;
  color: var(--azafran) !important; margin: 1.1rem 0 .2rem;
}
.estado .pie { font-size: .85rem; line-height: 1.5; opacity: .75; margin: .6rem 0 0; }
.estado code {
  display: block; margin-top: .6rem; padding: .6rem .7rem; border-radius: 8px;
  background: rgba(0,0,0,.25); color: var(--arroz) !important; font-size: .8rem;
  white-space: normal; word-break: break-all;
}

/* ---------- Resultados: la mesa ---------- */
.mesa-cab {
  display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
  flex-wrap: wrap; margin: 2.5rem 0 1.75rem; padding-top: 1.5rem;
  border-top: 1px solid var(--hoja);
}
.mesa-cab h2, .mesa-cab h2 span {
  font-family: var(--serif); font-weight: 400 !important; font-size: 1.75rem !important;
  line-height: 1.2; margin: 0; padding: 0; color: var(--albahaca) !important;
}
.mesa-cab span { color: var(--salvia); font-size: .95rem; }

.mesa {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 2.75rem 2rem;
  list-style: none; margin: 0; padding: 0;
}
@media (max-width: 900px) { .mesa { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .mesa { grid-template-columns: 1fr; } }

@property --p { syntax: '\\3C number\\3E'; inherits: false; initial-value: 0; }

.plato { display: flex; flex-direction: column; align-items: center; text-align: center; }
.anillo {
  --p: var(--meta);
  width: min(100%, 240px); aspect-ratio: 1; border-radius: 50%; padding: 7px;
  background: conic-gradient(var(--azafran) calc(var(--p) * 1%), var(--hoja) 0);
  animation: servir 1.1s cubic-bezier(.2,.7,.2,1) both;
  animation-delay: calc(var(--i) * 90ms);
}
@keyframes servir { from { --p: 0; } }
.anillo, .borde { box-sizing: border-box; }
.borde {
  width: 100%; aspect-ratio: 1; border-radius: 50%; padding: 11px;
  background: var(--plato);
  box-shadow: inset 0 0 0 1px #EEF1EB, 0 10px 24px -14px rgba(23,51,42,.45);
}
.borde img {
  width: 100%; height: auto !important; aspect-ratio: 1; object-fit: cover; border-radius: 50%; display: block;
}
.plato .nombre {
  font-family: var(--serif); font-size: 1.3rem; line-height: 1.2;
  margin: 1.1rem 0 .25rem; color: var(--albahaca);
}
.plato .dato { color: var(--salvia); font-size: .9rem; font-variant-numeric: tabular-nums; }
.plato .dato b { color: var(--albahaca); font-weight: 600; }

.consulta-img .anillo { --p: 0; animation: none; background: var(--albahaca); }

.vacio {
  margin-top: 2.5rem; padding: 2rem; border-radius: 24px; background: var(--hoja);
  max-width: 36rem;
}
.vacio p { margin: 0; line-height: 1.55; }
.vacio p + p { color: var(--salvia); margin-top: .4rem; }

@media (prefers-reduced-motion: reduce) {
  .anillo { animation: none; }
}
</style>
"""


def estilos() -> None:
    """Inyecta la hoja de estilos. Llamar al inicio de cada vista."""
    # Ojo: el sanitizador descarta todo el <style> si aparece un "<" literal
    # dentro del CSS; por eso @property usa la secuencia escapada \3C.
    st.html(_CSS)


def encabezado(titulo: str, bajada: str) -> None:
    st.markdown(
        f"<header class='hero'><h1>{escape(titulo)}</h1><p>{escape(bajada)}</p></header>",
        unsafe_allow_html=True,
    )


def nombre_plato(label: str) -> str:
    return label.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Estado del backend
# ---------------------------------------------------------------------------
def api_viva() -> dict | None:
    try:
        r = requests.get(f"{API}/health", timeout=3)
        if r.ok:
            return r.json()
    except requests.RequestException:
        return None
    return None


def barra_estado() -> bool:
    """Muestra el estado del backend en la barra lateral. True si esta listo."""
    est = api_viva()
    if est is None:
        cuerpo = (
            "<span class='luz' style='--c:#E5484D'>Backend apagado</span>"
            "<p class='pie'>Levanta la API en otra terminal:</p>"
            "<code>uvicorn api.main:app --port 8000</code>"
        )
        listo = False
    elif not est.get("listo"):
        cuerpo = (
            "<span class='luz' style='--c:#F0A202'>Índice no cargado</span>"
            "<p class='pie'>Genera el índice y reinicia la API:</p>"
            "<code>python construir_indice.py</code>"
        )
        listo = False
    else:
        cuerpo = (
            "<span class='luz' style='--c:#5BD08A'>Cocina abierta</span>"
            f"<p class='cifra'>{est['vectores']:,}</p>"
            f"<p class='pie' style='margin-top:0'>fotos de {est['clases']} platos en el índice</p>"
            f"<p class='pie'>Modelo: {escape(est['modelo'])}<br>Pesos: {escape(est['pesos'])}</p>"
            f"<p class='pie' style='margin-top:1rem'>{est['dim']} dimensiones, en {escape(est['dispositivo'])}</p>"
        )
        listo = True

    with st.sidebar:
        st.markdown(
            f"<section class='estado'><h3>Estado del sistema</h3>{cuerpo}</section>",
            unsafe_allow_html=True,
        )
    return listo


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
def plato_consulta(src: str, texto: str) -> str:
    """HTML de un plato sin anillo de similitud (la imagen de consulta)."""
    return (
        "<div class='plato consulta-img'><div class='anillo' style='--meta:0;--i:0'>"
        f"<div class='borde'><img src='{src}' alt='{escape(texto)}'></div></div>"
        f"<p class='nombre'>{escape(texto)}</p></div>"
    )


def mostrar_resultados(data: dict) -> None:
    """Sirve los resultados como platos; el arco azafrán marca la similitud
    relativa al mejor resultado de la búsqueda."""
    res = data.get("resultados", [])
    if not res:
        st.markdown(
            "<div class='vacio'><p>No encontramos platos para esa búsqueda.</p>"
            "<p>Prueba con el nombre del plato en inglés o con menos palabras.</p></div>",
            unsafe_allow_html=True,
        )
        return

    mejor = max(r["score"] for r in res) or 1.0
    platos = []
    for i, r in enumerate(res):
        nombre = escape(nombre_plato(r["label"]))
        meta = max(4.0, min(100.0, r["score"] / mejor * 100))
        platos.append(
            f"<li class='plato'>"
            f"<div class='anillo' style='--meta:{meta:.1f};--i:{i}' "
            f"title='Similitud {r['score']:.4f}'>"
            f"<div class='borde'><img src='{API}{escape(r['url'])}' alt='{nombre}'></div>"
            f"</div>"
            f"<p class='nombre'>{nombre}</p>"
            f"<p class='dato'><b>#{i + 1}</b>&nbsp; similitud {r['score']:.3f}</p>"
            f"</li>"
        )

    st.markdown(
        "<div class='mesa-cab'>"
        f"<h2>{data['n']} platos parecidos</h2>"
        f"<span>Encontrados en {data['ms']} ms</span></div>"
        f"<ol class='mesa'>{''.join(platos)}</ol>",
        unsafe_allow_html=True,
    )
