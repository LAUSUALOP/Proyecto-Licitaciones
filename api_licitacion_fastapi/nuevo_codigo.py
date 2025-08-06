from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import re
from bs4 import BeautifulSoup

app = FastAPI()

class LeyesRequest(BaseModel):
    leyes: list[str]

@app.post("/estado_leyes")
async def estado_leyes(request: LeyesRequest):
    resultados = {}
    for referencia in request.leyes:
        codigo_ley = extraer_codigo_ley(referencia)
        if not codigo_ley:
            resultados[referencia] = "No se reconoce formato tipo número/año"
            continue
        estado = await buscar_estado_ley_google(codigo_ley)
        resultados[referencia] = estado
    return resultados

def extraer_codigo_ley(texto: str) -> str | None:
    match = re.search(r"\b\d{1,4}/\d{4}\b", texto)
    return match.group(0) if match else None

async def buscar_estado_ley_google(codigo: str) -> str:
    try:
        params = {
            "engine": "google",
            "q": f"{codigo} site:boe.es",
            "api_key": "a4d98173a5e90bb141f98a4ddb379ee847a163dfed592f7be9e484fa8fcfa026KEY"
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://serpapi.com/search", params=params)
            data = resp.json()

            if "organic_results" not in data or not data["organic_results"]:
                return f"No encontrada en Google"

            url_boe = data["organic_results"][0]["link"]
            detalle = await client.get(url_boe)
            soup = BeautifulSoup(detalle.text, "html.parser")

            texto_en_rojo = soup.find_all("span", style=lambda x: x and "color:red" in x)
            for span in texto_en_rojo:
                if "derogada" in span.text.lower():
                    return "Derogada 🚫"

            return "Vigente ✅"
    except Exception as e:
        return f"Error: {str(e)}"


