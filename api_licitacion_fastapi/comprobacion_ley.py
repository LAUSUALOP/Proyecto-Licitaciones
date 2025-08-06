from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import urllib.parse
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
headers = {"User-Agent": "Mozilla/5.0"}

class Ley(BaseModel):
    nombre: str

class ListaLeyes(BaseModel):
    leyes: list[Ley]

def buscar_y_comprobar_ley(nombre_ley: str) -> dict:
    intentos = [
        nombre_ley,
        nombre_ley.split(",")[0],  # Antes de la coma
        " ".join(nombre_ley.split()[:3]),  # Las 3 primeras palabras
    ]

    for intento in intentos:
        try:
            query = urllib.parse.quote(intento)
            url_busqueda = f"https://www.boe.es/buscar/legislacion.php?q={query}"
            resp = requests.get(url_busqueda, headers=headers)

            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            enlaces = soup.select("li.resultado-busqueda a.resultado-busqueda-link-defecto")

            if not enlaces:
                continue

            href = enlaces[0].get("href")
            if not href:
                continue

            url_ley = f"https://www.boe.es{href}"
            resp_detalle = requests.get(url_ley, headers=headers)

            if resp_detalle.status_code != 200:
                continue

            texto = resp_detalle.text.lower()

            if "disposición derogada" in texto or "queda derogado" in texto:
                estado = "⚠️ Derogada"
            elif "vigente" in texto:
                estado = "✅ Vigente"
            else:
                estado = "❓ Indeterminado"

            return {"nombre": nombre_ley, "estado": estado, "url": url_ley}

        except Exception as e:
            logging.error(f"❌ Error con intento '{intento}' para '{nombre_ley}': {e}")
            continue

    return {"nombre": nombre_ley, "estado": "❓ No se encontró"}

@app.post("/estado_varias_leyes")
def estado_varias_leyes(data: ListaLeyes):
    resultados = [buscar_y_comprobar_ley(ley.nombre) for ley in data.leyes]
    return {"resultados": resultados}
