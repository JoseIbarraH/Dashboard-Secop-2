FROM python:3.12-slim

# Sin buffers en stdout (los logs salen en tiempo real) y sin .pyc en la imagen.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Dependencias primero: esta capa se reutiliza mientras requirements.txt no cambie.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Código de la app y su tema. El tema no es decorativo: sin él, Streamlit
#    sigue la preferencia del sistema y quien tenga el modo oscuro activado
#    recibe el encabezado en azul oscuro sobre negro, ilegible.
COPY pipeline.py lecturas.py app.py ./
COPY .streamlit ./.streamlit

# 3) (OPCIONAL) Incrustar el dataset para no depender de la red en tiempo de ejecución.
#    Si descomentas esto, sube también secop2_limpio.parquet al repo de la app.
#    Si lo dejas comentado, la app lo descarga sola desde GitHub al arrancar.
# COPY secop2_limpio.parquet .

# 4) Usuario sin privilegios: si alguien logra ejecutar código dentro del
#    contenedor, no lo hace como root. Streamlit necesita un HOME donde escribir
#    su configuración, de ahí el directorio propio.
RUN useradd --create-home --uid 10001 observatorio \
    && chown -R observatorio:observatorio /app
USER observatorio
ENV HOME=/home/observatorio

# Streamlit escucha en 8501 (este es el puerto que debes indicar en Dokploy).
EXPOSE 8501

# Chequeo de salud contra el endpoint que Streamlit expone. Se usa Python en vez
# de curl para no añadir paquetes de sistema solo para esto.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

# Arranque: bind a 0.0.0.0, sin abrir navegador, sin telemetría y sin vigilar
# cambios en los archivos (en producción el código no cambia, y el vigilante
# consume CPU y memoria para nada).
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
