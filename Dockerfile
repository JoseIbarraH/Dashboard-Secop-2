FROM python:3.12-slim

# curl solo para el healthcheck del contenedor
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencias (se cachean si no cambia requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Código de la app
COPY app.py .

# 3) (OPCIONAL) Incrustar el dataset para no depender de la red en tiempo de ejecución.
#    Si descomentas esto, sube también secop2_limpio.parquet al repo de la app.
#    Si lo dejas comentado, la app lo descarga sola desde tu GitHub al arrancar.
# COPY secop2_limpio.parquet .

# Streamlit escucha en 8501 (este es el puerto que debes indicar en Dokploy)
EXPOSE 8501

# Chequeo de salud que Streamlit expone en /_stcore/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Arranque: bind a 0.0.0.0, sin abrir navegador, sin telemetría
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
