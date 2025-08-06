from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup
import requests
import urllib.parse
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

headers = {"User-Agent": "Mozilla/5.0"}

class LeyConVariantes(BaseModel):
    nombres: List[str]

class ListaLeyesConVariantes(BaseModel):
    leyes: List[LeyConVariantes]

def buscar_y_comprobar_ley_varias_formas(lista_nombres: List[str]) -> dict:
    for nombre_ley in lista_nombres:
        try:
            query = urllib.parse.quote(nombre_ley)
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
            resp_ley = requests.get(url_ley, headers=headers)
            if resp_ley.status_code != 200:
                continue

            texto = resp_ley.text.lower()
            if "disposición derogada" in texto or "queda derogado" in texto:
                estado = "⚠️ Derogada"
            elif "vigente" in texto:
                estado = "✅ Vigente"
            else:
                estado = "❓ Indeterminado"

            return {"nombre_usado": nombre_ley, "estado": estado, "url": url_ley}

        except Exception as e:
            logging.error(f"Error buscando '{nombre_ley}': {e}")
            continue

    return {"nombre_usado": lista_nombres[0], "estado": "❓ No se encontró"}

@app.post("/estado_varias_leyes")
def estado_varias_leyes(data: ListaLeyesConVariantes):
    resultados = [buscar_y_comprobar_ley_varias_formas(ley.nombres) for ley in data.leyes]
    return {"resultados": resultados}

