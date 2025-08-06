#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
import qdrant_client
from qdrant_client.http import models as qmodels
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableMap


# ────────────────────────────────────────────────────────────────────────────────
# 0. Variables de entorno
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY        = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")

COLLECTION_NAME = "Prevencion_de_blanqueo_y_finanzas_sensibles"

# ─── CAMBIO ─── directorio de PDFs (no JSON)
PDF_DIR = "/home/reboot-student/Desktop/Licitacion/docs/leyes/a"

emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTIONS_SEARCH = [
    " Normativa General y Contratación Pública",
    "proteccion_de_datos_y_seguridad_digital",
    "sostenibilidad_recuperacion_y_fondos_europeos"
]

vectorstores = {
    name: Qdrant(
        client=client,
        collection_name=name,
        embeddings=emb
    )
    for name in COLLECTIONS_SEARCH
}
def contexto_multi_colecciones(query: str, k: int = 10) -> str:
    """
    Recorre cada colección, extrae los k documentos más similares
    y devuelve un string combinando todo, marcado por cabeceras.
    """
    bloques = []
    for nombre, store in vectorstores.items():
        docs = store.similarity_search(query, k=k)
        if not docs:
            continue
        # ­┆ Opcional: podrías resumir aquí cada bloque con el LLM si pesa demasiado
        bloque = "\n".join(d.page_content for d in docs)
        bloques.append(f"### {nombre}\n{bloque}")
    return "\n\n".join(bloques)


# ─── thresholds.py ───
EU_THRESHOLDS_2025 = {
     "services_subcentral": 221_000,   # €  Sin IVA
 }

SIMPLIFIED_LIMIT = 215_000           # hasta 214.999,99 € → art.159.1
ABBREV_LIMIT     = 60_000            # art.159.6


# ─── helpers_contract.py ───

def procedure_type(vec: float) -> str:
    if vec >= EU_THRESHOLDS_2025["services_subcentral"]:
        return "abierto_ordinario"
    if vec <= ABBREV_LIMIT:
        return "simplificado_abreviado"
    if vec <= SIMPLIFIED_LIMIT:
        return "abierto_simplificado"
    return "abierto_ordinario"

def sara(vec: float) -> bool:
    return vec >= EU_THRESHOLDS_2025["services_subcentral"]


# ────────────────────────────────────────────────────────────────────────────────
# 1. Historial
# ────────────────────────────────────────────────────────────────────────────────
history = InMemoryChatMessageHistory()
def construir_historial_chat(msgs):
    partes = []
    for m in msgs[-6:]:
        pref = "Humano: " if isinstance(m, HumanMessage) else "Asistente: "
        partes.append(pref + m.content)
    return "\n".join(partes)

# ────────────────────────────────────────────────────────────────────────────────
# 2‑3. LLM base y chat casual
# ────────────────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
chat_chain = (
    ChatPromptTemplate.from_messages(
        [("system", "Eres un asistente de IA amistoso y útil."),
         ("human", "{question}")]
    )
    | llm
)

# ────────────────────────────────────────────────────────────────────────────────
# 4. Carga y troceo de artículos DESDE PDF
# ────────────────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader            # ─── CAMBIO ───

# ─── CAMBIO ───: patrones y helpers para dividir PDFs en artículos
REG_TITULO_LINEA = re.compile(
    r'^\s*(?:Artículo\s+\d+[\.\)\-:]?\s+.+?|'
    r'Disposición\s+(?:adicional|transitoria|derogatoria|final)\s+\d*[A-Z]?[\.\)\-:]?.*?|'
    r'Preámbulo|Exposición\s+de\s+motivos|Anexo\s+\w+.*?)\s*$',
    re.IGNORECASE
)
REG_ARTICULO_INLINE = re.compile(
    r'\bArtículo\s+\d+[\.\)\-:]?\s+[A-ZÁÉÍÓÚÜÑ][^\n]*',
    re.IGNORECASE
)

def split_inline_articulos(texto, pagina, titulo_padre):
    mats = list(REG_ARTICULO_INLINE.finditer(texto))
    if not mats:
        return [{"titulo": titulo_padre or "(sin título)",
                 "contenido": texto.strip(), "pagina": pagina}]
    bloques = []
    for i, m in enumerate(mats):
        inicio = m.end()
        fin    = mats[i+1].start() if i+1<len(mats) else len(texto)
        titulo = titulo_padre if i==0 and titulo_padre else m.group(0).strip()
        bloques.append({"titulo": titulo,
                        "contenido": texto[inicio:fin].strip(),
                        "pagina": pagina})
    return bloques

def procesar_pdf(ruta_pdf: Path):
    loader  = PyPDFLoader(str(ruta_pdf))
    paginas = loader.load()

    bloques, bloque = [], {"titulo": None, "contenido": "", "pagina": None}

    for doc in paginas:
        pagina = doc.metadata.get("page") or doc.metadata.get("page_number")
        for linea in (doc.page_content or "").splitlines():
            linea = linea.strip()
            if not linea: 
                continue
            if REG_TITULO_LINEA.match(linea):
                if bloque["titulo"] and bloque["contenido"].strip():
                    bloques.extend(
                        split_inline_articulos(bloque["contenido"],
                                               bloque["pagina"],
                                               bloque["titulo"])
                    )
                bloque = {"titulo": linea, "contenido": "", "pagina": pagina}
            else:
                bloque["contenido"] += linea + "\n"

    if bloque["titulo"] and bloque["contenido"].strip():
        bloques.extend(
            split_inline_articulos(bloque["contenido"],
                                   bloque["pagina"],
                                   bloque["titulo"])
        )
    return bloques

# ─── FIN helpers PDF ───

def split_articulo_en_partes(titulo, texto, pagina, max_chars=1800):
    buffer, partes, idx = f"{titulo}\n", [], 1
    for p in texto.splitlines():
        p = p.strip()
        if not p: continue
        if len(buffer) + len(p) < max_chars:
            buffer += p + "\n"
        else:
            partes.append(
                Document(
                    page_content=f"[Página {pagina}]\n{titulo}\n(Parte {idx})\n{buffer.strip()}",
                    metadata={"titulo": titulo, "parte": idx, "page": pagina}
                )
            )
            idx, buffer = idx + 1, p + "\n"
    if buffer.strip():
        partes.append(
            Document(
                page_content=f"[Página {pagina}]\n{titulo}\n(Parte {idx})\n{buffer.strip()}",
                metadata={"titulo": titulo, "parte": idx, "page": pagina}
            )
        )
    return partes

# ─── CAMBIO ───: carga chunks directamente desde los PDF
def cargar_chunks_desde_pdfs(carpeta):
    chunks = []
    for path in Path(carpeta).glob("*.pdf"):
        for entrada in procesar_pdf(path):
            for p in split_articulo_en_partes(
                entrada["titulo"], entrada["contenido"], entrada["pagina"]
            ):
                p.metadata["source"] = path.name
                chunks.append(p)
    return chunks

def agrupar_chunks_por_titulo(chs):
    grupos = defaultdict(list)
    for d in chs:
        grupos[d.metadata["titulo"]].append(d)
    fusion = []
    for titulo, docs in grupos.items():
        docs.sort(key=lambda d: d.metadata.get("parte", 0))
        fusion.append(
            Document(
                page_content="\n".join(d.page_content for d in docs),
                metadata=docs[0].metadata,
            )
        )
    return fusion

# ─── CAMBIO ───
chunks = cargar_chunks_desde_pdfs(PDF_DIR)

# Índice auxiliar →  número de artículo  → [chunks]
idx_por_num = defaultdict(list)
pat_num = re.compile(r"\b(\d+)\b")
for c in chunks:
    if m := pat_num.search(c.metadata["titulo"]):
        idx_por_num[m.group(1)].append(c)

# ────────────────────────────────────────────────────────────────────────────────
# 7. Embeddings y Qdrant
# ────────────────────────────────────────────────────────────────────────────────

# if client.collection_exists(COLLECTION_NAME):
#     client.delete_collection(COLLECTION_NAME)

# client.create_collection(
#     COLLECTION_NAME,
#     vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE),
# )
vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=emb)

# vectorstore.add_documents(chunks)
# ────────────────────────────────────────────────────────────────────────────────
# 8. Cadena RAG
# ────────────────────────────────────────────────────────────────────────────────
rag_prompt = PromptTemplate.from_template(
    "Eres un experto redactor de licitaciones públicas en España, especializado en la elaboración de pliegos de cláusulas administrativas particulares. "
    "Tu tarea es redactar, desde cero, un pliego completo y técnicamente riguroso para un contrato de servicios informáticos (CPV 72500000). "
    "Debes utilizar un lenguaje jurídico-administrativo preciso, estructurado conforme a la normativa vigente en España.\n\n"
    "❗ No uses encabezados como 'CAPÍTULO' o 'Artículo'. Utiliza títulos o subtítulos naturales sin numeraciones rígidas ni estructura normativa.\n\n"
    "Debes fundamentarte especialmente en la Ley 9/2017, de 8 de noviembre, de Contratos del Sector Público, y citar los artículos relevantes "
    "en cada apartado (por ejemplo: objeto del contrato, procedimiento de adjudicación, criterios de solvencia, garantías, etc.). "
    "También puedes apoyarte en otras normas complementarias o guías oficiales cuando sea adecuado.\n\n"
    "Utiliza la documentación proporcionada como soporte y referencia directa, y ten en cuenta el historial del chat para mantener la coherencia del contenido.\n\n"
    "Historial del chat:\n{chat_history}\n\n"
    "Documentación de apoyo:\n{context}\n\n"
    "Instrucción o pregunta específica:\n{question}"
)
doc_chain = LLMChain(llm=llm, prompt=rag_prompt)
qa_chain = RunnableMap({
    "context":     lambda x: vectorstore.similarity_search(x["query"], k=25),
    "question":    lambda x: x["query"],
    "chat_history":lambda x: x["chat_history"],
}) | doc_chain

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt).replace('.', '').replace(',', '.'))
        except ValueError:
            print("❗ Introduce un número válido.")

def preguntar_datos():
    print("📝 Vamos a generar una licitación personalizada. Contesta las siguientes preguntas:\n")

    datos = {}

    # ── Pregunta preliminar ──
    datos["solo_software"] = input("¿Solo software (s/n)? ").lower().startswith("s")

    # 1. Datos del contrato
    print("🟢 1. DATOS DEL CONTRATO")
    datos["objeto_contrato"]       = input("¿Cuál es el objeto del contrato? ")
    datos["necesidad_resuelta"]    = input("¿Qué necesidad resuelve este servicio? ")
    datos["responsable_contrato"]  = input("¿Quién será el responsable del contrato (nombre y cargo)? ")
    datos["lugar_prestacion"]      = input("¿Dónde se prestará el servicio? ")

    # 2. Presupuesto
    print("\n💰 2. PRESUPUESTO")
    datos["pbl_sin_iva"]    = ask_float("¿PBL sin IVA (€)? ")
    datos["iva"]            = ask_float("¿% IVA aplicable? ")
    datos["duracion_meses"] = int(ask_float("Duración inicial (meses): "))
    datos["prorrogas"]      = int(ask_float("¿Meses de prórroga previstos? "))

    # 👉 Calcula VEC e infiere procedimiento y SARA
    vec = datos["pbl_sin_iva"] * (datos["duracion_meses"] + datos["prorrogas"]) / datos["duracion_meses"]

    datos.update({
        "vec":          vec,
        "procedimiento": procedure_type(vec),
        "sara":          sara(vec),
    })

    # 3. Empresa licitadora
    print("\n📄 3. EMPRESA LICITADORA")
    datos["nombre_empresa"]    = input("¿Nombre de la empresa? ")
    datos["cif"]               = input("¿CIF? ")
    datos["domicilio_fiscal"]  = input("¿Domicilio fiscal? ")
    datos["persona_contacto"]  = input("¿Nombre y correo de la persona de contacto? ")
    datos["censo_aeat"]        = input("¿Está inscrita la empresa en el Censo de empresarios de la AEAT? ")

    # 4. Documentación
    print("\n🔐 4. DOCUMENTACIÓN Y REQUISITOS")
    datos["declaracion_responsable"] = input("¿Presentarás la Declaración Responsable? ")
    datos["oferta_economica"]        = input("¿Tienes oferta económica lista? ")
    datos["acepta_pliego"]           = input("¿Aceptas las condiciones del pliego técnico? ")
    datos["perfiles_equipo"]         = input("¿Tu equipo cumple los perfiles mínimos exigidos? ")

    # 5. Protección de datos
    print("\n🔒 5. PROTECCIÓN DE DATOS")
    datos["trata_datos"] = input("¿Tratarás datos personales por cuenta del contratante? ").strip().lower()
    if datos["trata_datos"] in ["sí", "si", "s"]:
        datos["subcontrata_tratamiento"] = input("¿Subcontratarás servidores o servicios de tratamiento? ")
    else:
        datos["subcontrata_tratamiento"] = None

    # 6. Subcontratación
    print("\n📦 6. SUBCONTRATACIÓN")
    datos["subcontratacion"] = input("¿Vas a subcontratar alguna parte del servicio? ").strip().lower()
    if datos["subcontratacion"] in ["sí", "si", "s"]:
        datos["empresas_vinculadas"] = input("¿Tus subcontratistas son empresas no vinculadas? ")
    else:
        datos["empresas_vinculadas"] = None

    # 7. Criterios de valoración
    print("\n📉 7. CRITERIOS DE VALORACIÓN")
    datos["precio_ofertado"] = ask_float("¿Cuál es tu precio ofertado (sin IVA)? ")
    datos["precio_anormal"] = input("¿Has revisado que no sea anormalmente bajo? ")

    # 8. PRTR / NextGen
    print("\n🧾 8. NEXTGENERATION / PRTR")
    datos["cumple_prtr"] = input("¿Cumples con principios medioambientales y antifraude del PRTR? (sí/no): ").strip().lower()
    if datos["cumple_prtr"] in ["sí", "si", "s"]:
        datos["modelos_b1_b2_c"] = input("¿Has rellenado los modelos B1, B2 y C? ")
        datos["titular_real"]    = input("¿Quién es el titular real de la empresa? ")
    else:
        datos["modelos_b1_b2_c"] = None
        datos["titular_real"]    = None

    # 🛡️ GARANTÍAS
    print("\n🛡️  GARANTÍAS")
    datos["garantia_provisional"] = input("¿Exigir garantía provisional? (s/n): ").lower().startswith("s")
    if datos["garantia_provisional"]:
        datos["garantia_provisional_pct"] = ask_float("¿% o cuantía de esa garantía?: ")

    # 📊 SOLVENCIA
    print("\n📊 SOLVENCIA")
    datos["vol_negocios_min"]         = ask_float("Volumen anual de negocios mínimo (€): ")
    datos["servicios_similares_min"]  = ask_float("Importe anual servicios similares mínimo (€): ")
    datos["seguro_rc"]                = ask_float("Seguro de RC profesional mínimo (€): ")

    # 📉 9. PONDERACIÓN DE CRITERIOS
    print("\n📉 9. PONDERACIÓN DE CRITERIOS")
    datos["peso_metodologia"] = ask_float("% Metodología y plan de trabajo: ")
    datos["peso_equipo"]      = ask_float("% Equipo y experiencia: ")
    datos["peso_dnsk"]        = ask_float("% DNSH / sostenibilidad: ")
    datos["peso_precio"]      = ask_float("% Oferta económica: ")

    return datos


def generar_licitacion(datos):
    print("\n📄 RESUMEN DE LICITACIÓN GENERADO:\n")

    texto = f"""
🔹 Objeto del contrato:
{datos['objeto_contrato']}

🔹 Necesidad que se pretende resolver:
{datos['necesidad_resuelta']}

🔹 Lugar de prestación del servicio:
{datos['lugar_prestacion']}

🔹 Persona responsable del contrato:
{datos['responsable_contrato']}

🔹 Presupuesto base (sin IVA): {datos['pbl_sin_iva']} €
🔹 IVA aplicable: {datos['iva']}%
🔹 Prórrogas previstas: {datos['prorrogas']}

🔹 Empresa licitadora:
Nombre: {datos['nombre_empresa']}
CIF: {datos['cif']}
Domicilio fiscal: {datos['domicilio_fiscal']}
Persona de contacto: {datos['persona_contacto']}
Inscrita en el censo AEAT: {datos['censo_aeat']}

🔹 Documentación y requisitos:
- Declaración Responsable: {datos['declaracion_responsable']}
- Oferta Económica: {datos['oferta_economica']}
- Aceptación del Pliego Técnico: {datos['acepta_pliego']}
- Perfiles del equipo: {datos['perfiles_equipo']}
"""

    # Protección de datos (condicional)
    if datos["trata_datos"] in ["sí", "si", "s"]:
        texto += f"""
🔹 Protección de datos:
- ¿Tratará datos personales?: Sí
- ¿Subcontratará tratamiento?: {datos['subcontrata_tratamiento']}
"""
    else:
        texto += "\n🔹 Protección de datos: No se tratarán datos personales.\n"

    # Subcontratación (condicional)
    if datos["subcontratacion"] in ["sí", "si", "s"]:
        texto += f"""
🔹 Subcontratación:
- ¿Se subcontrata parte del servicio?: Sí
- Empresas no vinculadas: {datos['empresas_vinculadas']}
"""
    else:
        texto += "\n🔹 Subcontratación: No se prevé subcontratación.\n"

    # Oferta económica
    texto += f"""
🔹 Oferta económica:
- Precio ofertado: {datos['precio_ofertado']} €
- Revisión de precio anormalmente bajo: {datos['precio_anormal']}
"""

    # PRTR (condicional)
    if datos["cumple_prtr"] in ["sí", "si", "s"]:
        texto += f"""
🔹 Financiación NextGeneration / PRTR:
- Cumple principios PRTR: Sí
- Modelos B1, B2 y C: {datos['modelos_b1_b2_c']}
- Titular real de la empresa: {datos['titular_real']}
"""

    print(texto)

def construir_pregunta_final(respuestas):
    texto = "Quiero generar una licitación pública con los siguientes datos:\n\n"

    for clave, valor in respuestas.items():
        texto += f"- {clave.replace('_', ' ').capitalize()}: {valor}\n"

    texto += "\nRedacta la licitación completa con lenguaje técnico y formato oficial."
    return texto



if __name__ == "__main__":
    respuestas = preguntar_datos()
    generar_licitacion(respuestas) 
    query = construir_pregunta_final(respuestas)
    hist = construir_historial_chat(history.messages)
    MAX_MSGS = 100
    if len(history.messages) > MAX_MSGS:
        history.messages = history.messages[-MAX_MSGS:]
    contexto = contexto_multi_colecciones(query, k=25)   # ajusta k a tu gusto


    out = doc_chain.invoke({
        "question": query,
        "chat_history": hist,
        "context": contexto
    })

    answer = out["text"] if isinstance(out, dict) else out
    print("\n📄 LICITACIÓN GENERADA POR IA:\n")
    print(answer)

    history.add_user_message(query)
    history.add_ai_message(answer)

# ────────────────────────────────────────────────────────────────────────────────
# 9. Bucle principal interactivo
# ────────────────────────────────────────────────────────────────────────────────
print("🤖  Gemini RAG activo.  Escribe 'licitacion' para generar un pliego, 'salir' para terminar.\n")

while True:
    query = input("\nTú: ").strip()
    if not query:
        continue

    lower_q = query.lower()

    # --- Salida ---
    if lower_q == "salir":
        print("🫂  ¡Hasta luego!")
        break

    # --- Modo generación de licitación completa ---
    if lower_q == "licitacion":
        respuestas = preguntar_datos()
        query      = construir_pregunta_final(respuestas)

        contexto = contexto_multi_colecciones(query, k=10)
        hist     = construir_historial_chat(history.messages)

        out    = doc_chain.invoke({"question": query, "chat_history": hist, "context": contexto})
        answer = out["text"] if isinstance(out, dict) else out

        print("\n📄 LICITACIÓN GENERADA POR IA:\n")
        print(answer)

        history.add_user_message(query)
        history.add_ai_message(answer)

        # Poda opcional
        if len(history.messages) > 100:
            history.messages = history.messages[-100:]

        continue  # vuelve al prompt


    try:
        # --- Consulta normal ---
        # Si el usuario menciona un número que coincide con tu índice local por artículo,
        # prioriza ese match hiper-preciso de la colección recién cargada en memoria.
        num_match = re.search(r"\b(\d+)\b", query)
        if num_match and num_match.group(1) in idx_por_num:
            rel = idx_por_num[num_match.group(1)]
            rel.sort(key=lambda d: d.metadata.get("parte", 0))
            matches_docs = agrupar_chunks_por_titulo(rel)
            contexto = "\n\n".join(d.page_content for d in matches_docs)
        else:
            # Si no hay match directo → buscar en TODAS las colecciones Qdrant
            contexto = contexto_multi_colecciones(query, k=8)

        if contexto.strip():
            hist = construir_historial_chat(history.messages)
            out = doc_chain.invoke({"question": query, "chat_history": hist, "context": contexto})
            MAX_MSGS = 100
        if len(history.messages) > MAX_MSGS:
            history.messages = history.messages[-MAX_MSGS:]
            answer = out["text"] if isinstance(out, dict) else out

            print("\n📄  RAG:\n" + answer + "\n")

            history.add_user_message(query)
            history.add_ai_message(answer)
        else:
            # Fallback chat sin RAG
            resp = chat_chain.invoke({"question": query})
            print("\n🗣️  Chat:\n" + resp.content + "\n")
            history.add_user_message(query)
            history.add_ai_message(resp.content)

    except Exception as e:
        print("⚠️  Error en el loop:", e)
