"""
Frontend del buscador multimodal — Food-101.

Arrancar desde la raiz del proyecto, con la API ya corriendo:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Buscador de comida (Food-101)",
    page_icon="🍽️",
    layout="wide",
)

paginas = [
    st.Page("vistas/texto.py", title="Buscar por descripción", icon=":material/restaurant_menu:", default=True),
    st.Page("vistas/imagen.py", title="Buscar por foto", icon=":material/photo_camera:"),
]

st.navigation(paginas).run()
