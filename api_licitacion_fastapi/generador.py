from io import BytesIO
import markdown2
from weasyprint import HTML

def crear_pdf_desde_markdown(markdown_texto):
    # 1. Convertir Markdown a HTML
    html = markdown2.markdown(markdown_texto)
    
    # 2. Crear PDF desde HTML
    pdf_io = BytesIO()
    HTML(string=html).write_pdf(pdf_io)
    pdf_io.seek(0)
    
    return pdf_io



