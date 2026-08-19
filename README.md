# Observatorio ciudadano de contratación pública · Cartagena

App en Streamlit que convierte los datos abiertos del SECOP II (Distrito de
Cartagena) en un tablero interactivo: KPIs, panorama del gasto, priorización de
contratos por reglas y detección de anomalías con **Isolation Forest**.

## Archivos

| Archivo | Qué es |
|---|---|
| `pipeline.py` | Núcleo analítico: carga, señales de riesgo y modelo. No depende de Streamlit. |
| `lecturas.py` | Traduce los datos a lenguaje llano: qué es cada gráfica, qué dice y por qué importa. Tampoco depende de Streamlit. |
| `app.py` | La interfaz. |
| `tests/` | Pruebas del núcleo y del arranque de la app. |
| `requirements.txt` | Dependencias de ejecución, con versiones fijadas. |
| `requirements-dev.txt` | Lo anterior más `pytest`. |
| `.streamlit/config.toml` | Tema fijo (claro, azul institucional). No es decorativo: sin él, quien tenga el modo oscuro del sistema recibe el encabezado ilegible. |
| `Dockerfile` | Imagen para desplegar (ver `DESPLIEGUE_DOKPLOY.md`). |

## De dónde lee los datos

`pipeline.py` intenta, en este orden:

1. Un archivo local `secop2_limpio.parquet` junto a la app.
2. Si no está, lo descarga del repositorio de datos:
   `https://raw.githubusercontent.com/JoseIbarraH/datasets-analitica-de-datos/main/secop2_limpio.parquet`

Funciona tanto si subes el `.parquet` al repo de la app como si dejas que lo baje
del repositorio de datos. No hay que tocar código.

Si la descarga falla (sin red, timeout, ruta inexistente), la app muestra un
mensaje explicando qué pasó en vez de un error de programa.

### Variables de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `SECOP_DATA_REF` | `main` | Rama o **SHA de commit** del repositorio de datos. |
| `SECOP_DATA_URL` | — | URL completa alternativa; tiene prioridad sobre `SECOP_DATA_REF`. |
| `SECOP_DATA_TIMEOUT` | `60` | Segundos máximos de espera en la descarga. |

El repositorio de datos publica una actualización automática cada semana, así que
por defecto la app sigue los datos más recientes. Si vas a **respaldar una
publicación** y necesitas que los números no cambien, fija `SECOP_DATA_REF` al SHA
de un commit: la app queda anclada a esa versión exacta.

Sea cual sea la fuente, la app muestra en la barra lateral y en la pestaña de
metodología la **huella** (sha256 abreviado) del dataset en uso, para que
cualquier cifra publicada se pueda rastrear hasta los datos que la produjeron.

## Probar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

Las pruebas usan datos sintéticos: no necesitan descargar el dataset ni dependen
de que este no haya cambiado esta semana.

## Publicar gratis (Streamlit Community Cloud)

1. Sube `app.py`, `pipeline.py` y `requirements.txt` a un repositorio de GitHub (público).
   - Si quieres que la app lea los datos localmente, sube también
     `secop2_limpio.parquet` al mismo repo (pesa ~30 MB, cabe sin Git LFS).
   - Si no lo subes, la app los tomará automáticamente del repo de datos.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub.
3. "Create app" → elige el repositorio, la rama (`main`) y el archivo `app.py`.
4. "Deploy". En 1–2 minutos tendrás una **URL pública** del observatorio.

> Nota: no uses Git LFS para el `.parquet` en el repo de la app. Streamlit Cloud
> no resuelve punteros LFS y la lectura fallaría. Como archivo normal (<50 MB) va bien.

## El método, en corto

Los umbrales y el modelo se calculan sobre **todo** el universo de contratos; los
filtros de la interfaz solo eligen qué mostrar, nunca redefinen qué es atípico.

**Tres señales por reglas**, con cortes tomados de los propios datos:

1. **Valor alto** — contrato en el 5% superior (percentil 95).
2. **Contratación directa por encima de los pares** — la contratación directa es la
   norma en Cartagena (~84% de los contratos), así que marcarla a secas no
   distingue nada: señalaría a cinco de cada seis contratos. La señal se activa
   cuando la **entidad** que firma recurre a ella más que entidades comparables
   (percentil 75 entre las entidades con al menos 30 contratos). Las entidades sin
   ese volumen no se juzgan por esta vía.
3. **Adiciones de plazo extensas** — percentil 90 de las adiciones mayores que cero.

**Isolation Forest** (`contamination=0.02`, `n_estimators=200`, `random_state=42`)
sobre valor, duración, adiciones y modalidad. El valor entra en **escala
logarítmica**: sin transformar, domina la distancia y el modelo se limita a
redescubrir "los contratos más caros", que es justo lo que ya detecta la primera
regla.

Todos los parámetros están al principio de `pipeline.py` y se muestran en la
pestaña de metodología de la app.

## La capa de interpretación

El tablero está pensado para alguien que no sabe qué es una mediana ni una
modalidad de contratación. Por eso cada apartado tiene un icono 💡 **¿Qué
significa esto?** que abre una explicación de tres partes: qué estás viendo, qué
dicen **estos** datos concretos y por qué importa.

Esas explicaciones viven en `lecturas.py` y se **calculan a partir de los datos
filtrados**, no son textos fijos. Si el ciudadano filtra por una entidad, la
explicación habla de esa entidad y de sus cifras. Así no puede pasar que el texto
diga una cosa y la gráfica muestre otra.

Además hay un **resumen en palabras** al principio del panorama —lo que debería
bastar si alguien solo lee una cosa— y un **glosario** al final de la metodología.

Como `lecturas.py` es Python puro sobre un DataFrame, se puede probar (`pytest`
comprueba que las cifras citadas coinciden con los datos) y se puede reutilizar
desde cualquier otra interfaz.

## Buscar

La barra lateral abre con un buscador libre sobre **entidad, proveedor y número de
contrato**. No hace falta poner tildes ni mayúsculas, ni escribir el nombre
completo, y las palabras pueden ir en cualquier orden: «naval hospital» encuentra
lo mismo que «hospital naval».

Es la pregunta que más se hace quien entra a una herramienta así —«¿qué contratos
tiene esta empresa?»— y antes no se podía responder.

## Pensado para compartirse

Los filtros y la búsqueda viajan en la dirección de la página. Si alguien filtra por una entidad
y un año y encuentra algo que no cuadra, puede copiar el enlace y quien lo reciba
verá exactamente esa vista. En una herramienta de control ciudadano, poder decir
"mira esto" es media utilidad.

Además, cada contrato de las listas de priorización enlaza a su **ficha oficial en
el SECOP II**, para que nadie tenga que creerse este tablero: puede ir a
comprobarlo en la fuente.

## Encuadre responsable

Un contrato marcado como "atípico" es estadísticamente inusual, **no**
necesariamente irregular. La herramienta señala dónde mirar; no acusa.
