# Observatorio ciudadano de contratación pública · Cartagena

App en Streamlit que convierte los datos abiertos del SECOP II (Distrito de
Cartagena) en un tablero interactivo: KPIs, panorama del gasto, priorización de
contratos por reglas y detección de anomalías con **Isolation Forest**.

Reproduce exactamente la lógica del notebook `Secop_2_Graficas_Articulo_Definitivo.ipynb`
(mismos umbrales por percentiles y mismos parámetros del modelo:
`contamination=0.02`, `n_estimators=200`, `random_state=42`).

## Archivos

- `app.py` — la aplicación.
- `requirements.txt` — dependencias.
- `secop2_limpio.parquet` — dataset limpio (opcional aquí; ver más abajo).

## De dónde lee los datos

`app.py` intenta, en este orden:

1. Un archivo local `secop2_limpio.parquet` junto a `app.py`.
2. Si no está, lo descarga de tu GitHub:
   `https://raw.githubusercontent.com/JoseIbarraH/datasets-analitica-de-datos/main/secop2_limpio.parquet`

Es decir: funciona tanto si subes el `.parquet` al mismo repo que la app como si
dejas la app sola y que lo baje de tu repositorio de datos. No hay que tocar código.

## Probar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`.

## Publicar gratis (Streamlit Community Cloud)

1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub (público).
   - Si quieres que la app lea los datos localmente, sube también
     `secop2_limpio.parquet` al mismo repo (pesa ~30 MB, cabe sin Git LFS).
   - Si no lo subes, la app los tomará automáticamente de tu repo de datos.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub.
3. "Create app" → elige el repositorio, la rama (`main`) y el archivo `app.py`.
4. "Deploy". En 1–2 minutos tendrás una **URL pública** del observatorio.

> Nota: no uses Git LFS para el `.parquet` en el repo de la app. Streamlit Cloud
> no resuelve punteros LFS y la lectura fallaría. Como archivo normal (<50 MB) va bien.

## Encuadre responsable

Un contrato marcado como "atípico" es estadísticamente inusual, **no**
necesariamente irregular. La herramienta señala dónde mirar; no acusa.
