"""Interfaz A: busqueda por consulta textual (texto -> imagen)."""

import requests
import streamlit as st

from ui_comun import API, barra_estado, encabezado, estilos, mostrar_resultados

estilos()
listo = barra_estado()

EJEMPLOS = [
    "a plate of sushi",
    "chocolate cake with strawberries",
    "a bowl of hot ramen",
    "fried chicken and french fries",
    "pancakes with maple syrup",
    "tacos with guacamole",
]

if "consulta" not in st.session_state:
    st.session_state.consulta = st.query_params.get("q", "")


def _usar_ejemplo() -> None:
    if st.session_state.ejemplo:
        st.session_state.consulta = st.session_state.ejemplo


# Portada: encabezado, buscador y ejemplos sobre la foto de fondo.
with st.container(key="portada"):
    encabezado(
        "¿Qué se te antoja hoy?",
        "Describe un plato con tus palabras y te mostramos las fotos que más se le parecen "
        "entre 101 tipos de comida.",
    )

    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    consulta = c1.text_input(
        "Describe un plato",
        key="consulta",
        placeholder="Por ejemplo: a bowl of spicy noodles",
        label_visibility="collapsed",
    )
    buscar = c2.button("Buscar", type="primary", use_container_width=True)

    st.pills(
        "Ideas para empezar",
        EJEMPLOS,
        key="ejemplo",
        on_change=_usar_ejemplo,
        label_visibility="collapsed",
    )
    st.markdown(
        "<p class='nota'>El modelo entiende mejor las descripciones en inglés.</p>",
        unsafe_allow_html=True,
    )

if (buscar or consulta) and listo and consulta.strip():
    with st.spinner("Buscando platos..."):
        try:
            r = requests.post(
                f"{API}/search/text",
                json={"query": consulta.strip(), "top_k": 6},
                timeout=30,
            )
            r.raise_for_status()
            mostrar_resultados(r.json())
        except requests.RequestException as e:
            st.error(f"La API no respondió a la búsqueda. Revisa que siga corriendo. Detalle: {e}")
elif (buscar or consulta) and not listo:
    st.warning("El buscador aún no está listo. Revisa el estado del sistema en la barra lateral.")
