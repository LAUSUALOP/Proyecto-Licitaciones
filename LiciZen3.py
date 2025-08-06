#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json
from pathlib import Path
from collections import defaultdict
from hashlib import sha1
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
from langchain.memory import ConversationBufferMemory

# ────────────────────────────────────────────────────────────────────────────────
# 0. Variables de entorno
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY        = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")

COLLECTION_NAME = "RAG_Licitaciones"

# ─── CAMBIO ─── directorio de PDFs (no JSON)
PDF_DIR = "/home/reboot-student/Desktop/Licitacion/docs/leyes/ Normativa General y Contratación Pública/"

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
        pagina = doc.metadata.get("page")
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
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# if client.collection_exists(COLLECTION_NAME):
#     client.delete_collection(COLLECTION_NAME)

# client.create_collection(
#     COLLECTION_NAME,
#     vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE),
# )
vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=emb)

# vectorstore.add_documents(chunks)
# licitacion_bot.py
# ────────────────────────────────────────────────────────────────────────────────
# 8. Cadena RAG
# ────────────────────────────────────────────────────────────────────────────────
rag_prompt = PromptTemplate.from_template(
    "Eres un creador experto de licitaciones públicas en España. "
    "Crea con lenguaje técnico, completo y apoyándote en la documentación.\n\n"
    "Historial:\n{chat_history}\n\nDocumentación:\n{context}\n\n"
    "Pregunta: {question}"
)

doc_chain = LLMChain(llm=llm, prompt=rag_prompt)
qa_chain = RunnableMap({
    "context":     lambda x: vectorstore.similarity_search(x["query"], k=25),
    "question":    lambda x: x["query"],
    "chat_history":lambda x: x["chat_history"],
}) | doc_chain

def preguntar_datos():
    print("📝 Vamos a generar una licitación personalizada. Contesta las siguientes preguntas:\n")

    datos = {}

    # 1. Datos del contrato
    print("🟢 1. DATOS DEL CONTRATO")
    datos["objeto_contrato"] = input("¿Cuál es el objeto del contrato? ")
    datos["necesidad_resuelta"] = input("¿Qué necesidad resuelve este servicio? ")
    datos["responsable_contrato"] = input("¿Quién será el responsable del contrato (nombre y cargo)? ")
    datos["lugar_prestacion"] = input("¿Dónde se prestará el servicio? ")

    # 2. Presupuesto
    print("\n💰 2. PRESUPUESTO")
    datos["pbl_sin_iva"] = input("¿Cuál es el presupuesto base sin IVA? ")
    datos["iva"] = input("¿Cuál es el porcentaje de IVA aplicable? ")
    datos["prorrogas"] = input("¿Hay prórrogas previstas? ¿De cuántos meses? ")

    # 3. Empresa licitadora
    print("\n📄 3. EMPRESA LICITADORA")
    datos["nombre_empresa"] = input("¿Nombre de la empresa? ")
    datos["cif"] = input("¿CIF? ")
    datos["domicilio_fiscal"] = input("¿Domicilio fiscal? ")
    datos["persona_contacto"] = input("¿Nombre y correo de la persona de contacto? ")
    datos["censo_aeat"] = input("¿Está inscrita la empresa en el Censo de empresarios de la AEAT? ")

    # 4. Documentación
    print("\n🔐 4. DOCUMENTACIÓN Y REQUISITOS")
    datos["declaracion_responsable"] = input("¿Presentarás la Declaración Responsable? ")
    datos["oferta_economica"] = input("¿Tienes oferta económica lista? ")
    datos["acepta_pliego"] = input("¿Aceptas las condiciones del pliego técnico? ")
    datos["perfiles_equipo"] = input("¿Tu equipo cumple los perfiles mínimos exigidos? ")

    # 5. Protección de datos
    print("\n🔒 5. PROTECCIÓN DE DATOS")
    datos["trata_datos"] = input("¿Tratarás datos personales por cuenta del contratante? ")
    datos["subcontrata_tratamiento"] = input("¿Subcontratarás servidores o servicios de tratamiento? ")

    # 6. Subcontratación
    print("\n📦 6. SUBCONTRATACIÓN")
    datos["subcontratacion"] = input("¿Vas a subcontratar alguna parte del servicio? ")
    datos["empresas_vinculadas"] = input("¿Tus subcontratistas son empresas no vinculadas? ")

    # 7. Criterios de valoración
    print("\n📉 7. CRITERIOS DE VALORACIÓN")
    datos["precio_ofertado"] = input("¿Cuál es tu precio ofertado (sin IVA)? ")
    datos["precio_anormal"] = input("¿Has revisado que no sea anormalmente bajo? ")

    # 8. PRTR / NextGen
    print("\n🧾 8. NEXTGENERATION / PRTR")
    datos["modelos_b1_b2_c"] = input("¿Has rellenado los modelos B1, B2 y C? ")
    datos["titular_real"] = input("¿Quién es el titular real de la empresa? ")
    datos["cumple_prtr"] = input("¿Cumples con principios medioambientales y antifraude del PRTR? ")

    return datos
def generar_licitacion(datos):
    print("\n📄 LICITACIÓN GENERADA:\n")

    texto = f"""
1. OBJETO Y FINALIDAD DEL CONTRATO
Objeto del contrato: {datos['objeto_contrato']}
Necesidad que resuelve: {datos['necesidad_resuelta']}

2. LUGAR DE PRESTACIÓN Y RESPONSABLE
Lugar: {datos['lugar_prestacion']}
Responsable: {datos['responsable_contrato']}

3. PRESUPUESTO Y CONDICIONES ECONÓMICAS
Presupuesto base sin IVA: {datos['pbl_sin_iva']} €
IVA aplicable: {datos['iva']}%
Prórrogas previstas: {datos['prorrogas']}

4. IDENTIFICACIÓN DE LA EMPRESA LICITADORA
Nombre: {datos['nombre_empresa']}
CIF: {datos['cif']}
Domicilio fiscal: {datos['domicilio_fiscal']}
Persona de contacto: {datos['persona_contacto']}
Inscrita en el censo de AEAT: {datos['censo_aeat']}

5. DOCUMENTACIÓN Y COMPROMISOS
Declaración responsable: {datos['declaracion_responsable']}
Oferta económica lista: {datos['oferta_economica']}
Aceptación del pliego técnico: {datos['acepta_pliego']}
Equipo con perfiles adecuados: {datos['perfiles_equipo']}

6. PROTECCIÓN DE DATOS Y SUBCONTRATACIÓN
¿Tratará datos personales?: {datos['trata_datos']}
¿Subcontratará tratamiento?: {datos['subcontrata_tratamiento']}
¿Subcontrata parte del servicio?: {datos['subcontratacion']}
¿Empresas no vinculadas?: {datos['empresas_vinculadas']}

7. OFERTA ECONÓMICA Y CRITERIOS DE VALORACIÓN
Precio ofertado: {datos['precio_ofertado']} €
Precio no anormalmente bajo: {datos['precio_anormal']}

8. CUMPLIMIENTO DE NORMATIVA PRTR
Modelos B1, B2 y C presentados: {datos['modelos_b1_b2_c']}
Titular real de la empresa: {datos['titular_real']}
Cumplimiento PRTR: {datos['cumple_prtr']}
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

    query = construir_pregunta_final(respuestas)
    hist = construir_historial_chat(history.messages)
    contexto = "\n\n".join(d.page_content for d in vectorstore.similarity_search(query, k=25))

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



# ─── DEBUG ───
DEBUG_SHOW_CONTEXT = True
def resumen_docs(documents):
    lineas = []
    for d in documents:
        meta = d.metadata
        lineas.append(
            f"{meta.get('source','?')} | {meta.get('titulo')[:60]} "
            f"(parte {meta.get('parte','?')}, pág. {meta.get('page','?')})"
        )
    return "\n".join(lineas)

# ────────────────────────────────────────────────────────────────────────────────
# 9. Bucle principal
# ────────────────────────────────────────────────────────────────────────────────
print("🤖  Gemini RAG activo.  'salir' para terminar.\n")

while True:
    query = input("\nTú: ").strip()
    if query.lower() == "salir":
        print("🫂  ¡Hasta luego!"); 
        break

    if query.lower() == "licitacion":
        respuestas = preguntar_datos()
        query = construir_pregunta_final(respuestas)


        break

    try:
        num_match = re.search(r"\b(\d+)\b", query)
        if num_match and num_match.group(1) in idx_por_num:
            rel = idx_por_num[num_match.group(1)]
            rel.sort(key=lambda d: d.metadata.get("parte", 0))
            matches = agrupar_chunks_por_titulo(rel)
        else:
            vecinos = vectorstore.similarity_search(query, k=15)
            matches = agrupar_chunks_por_titulo(vecinos)

        if matches:
            hist = construir_historial_chat(history.messages)
            contexto = "\n\n".join(d.page_content for d in matches)

            out = doc_chain.invoke(
                {"question": query, "chat_history": hist, "context": contexto}
            )
            answer = out["text"] if isinstance(out, dict) else out
            print("\n📄  RAG:\n" + answer + "\n")

            history.add_user_message(query)
            history.add_ai_message(answer)
        else:
            resp = chat_chain.invoke({"question": query})
            print("\n🗣️  Chat:\n" + resp.content + "\n")
            history.add_user_message(query)
            history.add_ai_message(resp.content)

    except Exception as e:
        print("⚠️ ", e)