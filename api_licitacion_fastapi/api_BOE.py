import requests
from time import sleep

# Lista de leyes/normas que quieres comprobar
leyes = [
    "Ley 9/1968, de 5 de abril, sobre secretos oficiales",
    "Ley 9/2017, de 8 de noviembre, de Contratos del Sector Público",
    "Ley 11/2002, de 6 de mayo, reguladora del Centro Nacional de Inteligencia",
    "Ley 38/2003, de 17 de noviembre, General de Subvenciones",
    "Ley 39/2015, de 1 de octubre, del Procedimiento Administrativo Común",
    "Ley 40/2015, de 1 de octubre, de Régimen Jurídico del Sector Público",
    "Ley 19/2013, de 9 de diciembre, de transparencia, acceso a la información pública y buen gobierno",
    "Ley Orgánica 3/2018, de 5 de diciembre, de Protección de Datos Personales",
    "Norma Española UNE-EN ISO/IEC 27001 Mayo 2017",
    "Resolución de 19 de julio de 2011, Documento Electrónico",
    "Orden HPF/1030/2021",
    "Orden HPF/1031/2021",
    "Real Decreto 4/2010, de 8 de enero, Esquema Nacional de Interoperabilidad",
    "Real Decreto 311/2022, de 3 de mayo, Esquema Nacional de Seguridad",
    "Real Decreto 421/2004, de 12 de marzo, Centro Criptológico Nacional",
    "Real Decreto 817/2009, de 8 de mayo, desarrolla Ley 30/2007"
]

def comprobar_estado_ley(nombre_ley):
    url = f"https://www.boe.es/buscar/legislacion.php?q={nombre_ley}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Evita bloqueos por parte del BOE
    try:
        response = requests.get(url, headers=headers)
        texto = response.text.lower()

        if "derogada" in texto:
            return "⚠️ Derogada"
        elif "vigente" in texto:
            return "✅ Vigente"
        else:
            return "❓ No se pudo determinar"
    except Exception as e:
        return f"❌ Error: {e}"

# Comprobamos todas las leyes
for ley in leyes:
    estado = comprobar_estado_ley(ley)
    print(f"{ley} --> {estado}")
    sleep(1)  # Para no abusar del servidor del BOE