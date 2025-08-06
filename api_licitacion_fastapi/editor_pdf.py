import streamlit as st
from PIL import Image
import base64
import os
import requests

st.title("✍️ Editor Markdown con imágenes y firma")

# Editor
markdown_text = st.text_area("Escribe en Markdown aquí:", height=300)

# Subida de imágenes
uploaded_file = st.file_uploader("Sube una imagen o firma (PNG/JPG):", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Codificar en base64
    buffered = uploaded_file.read()
    encoded = base64.b64encode(buffered).decode()
    ext = os.path.splitext(uploaded_file.name)[-1].replace(".", "")
    img_tag = f'<img src="data:image/{ext};base64,{encoded}" width="300"/>'

    if st.button("Insertar imagen en el Markdown"):
        markdown_text += "\n\n" + img_tag

# Vista previa
st.markdown("---")
st.subheader("📄 Vista previa:")
st.markdown(markdown_text, unsafe_allow_html=True)

# --- Generar PDF llamando a la API
if st.button("📥 Generar PDF"):
    with st.spinner("Generando PDF..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/generar-pdf",
                json={"contenido_md": markdown_text}
            )
            if response.status_code == 200:
                pdf_bytes = response.content
                st.success("✅ PDF generado con éxito")
                st.download_button(
                    label="📄 Descargar PDF",
                    data=pdf_bytes,
                    file_name="documento.pdf",
                    mime="application/pdf"
                )
            else:
                st.error(f"❌ Error al generar PDF: {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Error de conexión: {e}")
