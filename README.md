# Análisis Automatizado de Documentos Legales y Financieros para la revisión de Mergers y Adquisitions
## Link https://agentesmerger.streamlit.app/
Este proyecto permite extraer, analizar y estructurar información clave de documentos PDF o archivos de texto (No acepta imagenes ni pdf que no esten de lectura) relacionados con:

* Comprobantes de depósitos bancarios
* Operaciones de fusión y adquisición
* Poderes legales
* Contratos legales

## Funcionalidades
📄 Extracción de Texto
Soporte para PDF, imágenes (OCR con Tesseract) y archivos .txt (Esto solo funciona si se corre localmente, desafortunadamente no he podido sustituir el OCR tesseract que no corre en Streamlit)
OCR automático cuando el PDF no contiene texto legible

🔍 Análisis de Depósitos
Extrae JSON con los campos: beneficiario, ordenante, monto, fecha_hora
Utiliza el modelo deepseek-chat para interpretar comprobantes y arroja la hora de pago, el que deposita y el que recibe el dinero

⚖️ Análisis de Fusiones
Extrae partes relevantes de una operación (compradores, vendedores, monto, traslapes, etc.)
Genera una narrativa ejecutiva con gpt-4-turbo (Se considero que es más apto que Deepseek, se hicieron varias pruebas y fue funcionalmente superior).
Permite generar un dictamen formal en formato .docx

🧾 Análisis de Poderes Legales
Identifica otorgante, apoderado, fecha y facultades a partir del texto del documento
Devuelve los datos en formato estructurado JSON

## Requisitos
Python 3.8+
Streamlit
Tesseract OCR
OpenAI API (GPT y/o DeepSeek)
Librerías: pdfplumber, pytesseract, Pillow, docx, re, json, tempfile, datetime, etc.

## Cómo usar
Ejecuta la aplicación con streamlit run app.py
* Sube un documento PDF, imagen o texto
* Selecciona el análisis deseado (depósito, fusión o poder)
* Visualiza los resultados y descarga el JSON o dictamen generado

## Créditos
Desarrollado para facilitar el análisis documental en contextos legales, financieros y regulatorios por mi. Una parte del código fue generada por Chatgpt (RAG), Deepseek (Streamlit), Mistral (Función poderes) y Anthropic (correciones y atención a errores). Esta la versión 3.0. La primera versión corría en Colab, la segunda versión incorporó la revisión de escritos iniciales. En total se usaron cerca de 40 prompts. 

## Prueba
En internet hay varios documentos legales que pueden ser usados de prueba, recomiendo:

* https://impeweb.mpiochih.gob.mx/transparencia/recursos_materiales/BAXTER/PODER%20LEGAL%20BAXTER.pdf
* https://oci.wi.gov/Documents/Companies/FinPacificIndEx1.pdf
