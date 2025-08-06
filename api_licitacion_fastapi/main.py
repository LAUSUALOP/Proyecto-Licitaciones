from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from generador import crear_pdf_desde_markdown

app = FastAPI()

class MarkdownEntrada(BaseModel):
    contenido_md: str

@app.post("/generar-pdf")
def generar_pdf(markdown: MarkdownEntrada):
    pdf_stream = crear_pdf_desde_markdown(markdown.contenido_md)
    
    return StreamingResponse(
        pdf_stream,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=markdown.pdf"}
    )



