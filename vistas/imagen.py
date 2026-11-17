"""Interfaz B: busqueda por imagen de consulta (imagen -> imagen)."""

import base64

import requests
import streamlit as st

from ui_comun import (
    API,
    barra_estado,
    encabezado,
    estilos,
    mostrar_resultados,
    plato_consulta,
)

estilos()
encabezado(
    "Sube una foto de tu plato",
    "Te mostramos los platos del índice que más se le parecen: mismo tipo de comida, "
    "colores o presentación.",
)

listo = barra_estado()

archivo = st.file_uploader(
    "Foto de comida (JPG, PNG, WebP o BMP, hasta 10 MB)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if archivo is not None:
    datos = archivo.getvalue()
    src = f"data:{archivo.type};base64,{base64.b64encode(datos).decode()}"

    c1, c2 = st.columns([1, 2], vertical_alignment="center", gap="large")
    with c1:
        st.markdown(plato_consulta(src, "Tu foto"), unsafe_allow_html=True)
    with c2:
        buscar = st.button("Buscar platos parecidos", type="primary", disabled=not listo)
        if not listo:
            st.caption("El buscador aún no está listo. Revisa el estado en la barra lateral.")

    if buscar and listo:
        with st.spinner("Buscando platos..."):
            try:
                r = requests.post(
                    f"{API}/search/image",
                    files={"file": (archivo.name, datos, archivo.type)},
                    data={"top_k": 6},
                    timeout=60,
                )
                r.raise_for_status()
                mostrar_resultados(r.json())
            except requests.RequestException as e:
                st.error(f"La API no pudo procesar la foto. Detalle: {e}")
