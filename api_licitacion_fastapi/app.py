import streamlit as st
import requests

# 👉 Pega aquí la URL que te da ngrok, sin la última barra
API_URL = "https://d59be2d72fce.ngrok-free.app"

st.set_page_config(page_title="Generador de Pliegos", layout="centered")
st.title("📄 Generador de Pliegos para Licitaciones")
st.markdown("Completa los datos del proyecto y genera los documentos automáticamente.")

# --- FORMULARIO ---
with st.form("formulario_pliegos"):
    nombre_proyecto = st.text_input("🔤 Nombre del Proyecto")
    tipo_contrato = st.selectbox("📌 Tipo de Contrato", ["Servicios", "Obras", "Suministros"])
    importe = st.number_input("💰 Importe Estimado (€)", step=1000)
    submitted = st.form_submit_button("🚀 Generar Pliegos")

# --- LLAMADA A LA API ---
if submitted:
    with st.spinner("Generando documentos..."):
        try:
            payload = {
                "nombre": nombre_proyecto,
                "tipo": tipo_contrato,
                "importe": importe
            }

            # POST a tu endpoint FastAPI
            response = requests.post(f"{API_URL}/generar", json=payload)

            if response.status_code == 200:
                data = response.json()
                st.success("✅ ¡Pliegos generados correctamente!")

                # Mostrar contenido en Markdown
                st.subheader("📑 Vista previa (Markdown)")
                st.markdown(data.get("markdown", "_Sin contenido disponible_"))

                # Botones de descarga de PDF
                st.download_button(
                    "📥 Descargar Pliego Técnico",
                    data["pliego_tecnico"].encode("utf-8"),
                    file_name="Pliego_Tecnico.pdf",
                    mime="application/pdf"
                )

                st.download_button(
                    "📥 Descargar Pliego Administrativo",
                    data["pliego_administrativo"].encode("utf-8"),
                    file_name="Pliego_Administrativo.pdf",
                    mime="application/pdf"
                )

            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"⚠️ Error al conectar con la API: {e}")
