# -*- coding: utf-8 -*-
"""
Núcleo analítico del observatorio de contratación pública de Cartagena.

Carga el dataset del SECOP II, deriva las variables de análisis, aplica las
señales de riesgo por reglas y entrena el detector de anomalías.

No importa Streamlit a propósito: así el método se puede ejecutar desde un
script, un notebook o la batería de pruebas sin levantar la interfaz.
"""
from __future__ import annotations

import hashlib
import io
import os
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Fuente de datos
# --------------------------------------------------------------------------- #
ARCHIVO_LOCAL = "secop2_limpio.parquet"
REPO_DATOS = "JoseIbarraH/datasets-analitica-de-datos"

# El repositorio de datos publica una actualización automática cada semana, así
# que por defecto seguimos su rama principal. Para respaldar una publicación con
# datos idénticos a los del día del análisis, fija SECOP_DATA_REF al SHA de un
# commit: la app queda anclada a esa versión exacta del dataset.
REF_DATOS = os.environ.get("SECOP_DATA_REF", "main")
URL_DATOS = os.environ.get("SECOP_DATA_URL") or (
    f"https://raw.githubusercontent.com/{REPO_DATOS}/{REF_DATOS}/{ARCHIVO_LOCAL}"
)
TIMEOUT_DESCARGA = int(os.environ.get("SECOP_DATA_TIMEOUT", "60"))

# --------------------------------------------------------------------------- #
# Parámetros del método (se muestran en la pestaña de metodología)
# --------------------------------------------------------------------------- #
PCT_VALOR_ALTO = 0.95        # un contrato "caro" es el 5% superior del universo
PCT_ADICIONES = 0.90         # sobre las adiciones estrictamente positivas
PCT_DIRECTA_PARES = 0.75     # entidad que usa directa más que sus pares
MIN_CONTRATOS_ENTIDAD = 30   # volumen mínimo para que la tasa de una entidad sea estable

CONTAMINATION = 0.02
N_ESTIMATORS = 200
RANDOM_STATE = 42
MIN_FILAS_MODELO = 50        # por debajo de esto el Isolation Forest no es informativo

RANGOS_VALOR = ["< 90 M", "90–500 M", "500 M–1 B", "> 1 B"]
_BINS_VALOR = [0, 90e6, 500e6, 1e9, float("inf")]
NIVELES_RIESGO = {0: "Bajo", 1: "Medio", 2: "Alto", 3: "Crítico"}

COLUMNAS_REQUERIDAS = (
    "Nombre Entidad",
    "ID Contrato",
    "Referencia del Contrato",
    "Modalidad de Contratacion",
    "Proveedor Adjudicado",
    "Es Pyme",
    "Valor del Contrato",
    "Dias adicionados",
    "Fecha de Firma",
    "Fecha de Inicio del Contrato",
    "Fecha de Fin del Contrato",
)

# Se leen si el dataset las trae, pero la app funciona sin ellas.
COLUMNA_URL = "urlproceso"
COLUMNAS_OPCIONALES = (COLUMNA_URL,)


class ErrorDeDatos(RuntimeError):
    """La fuente de datos no se pudo leer o no tiene la forma esperada."""


@dataclass(frozen=True)
class Fuente:
    """De dónde salieron los datos que se están mostrando."""

    origen: str
    huella: str      # sha256 abreviado: identifica la versión exacta del dataset
    n_bytes: int


# --------------------------------------------------------------------------- #
# Formato
# --------------------------------------------------------------------------- #
def fmt_cop(x) -> str:
    """Formatea pesos con la convención colombiana: $1,24 B / $1,89 mil M / $322 M."""
    try:
        if x is None or pd.isna(x) or not np.isfinite(float(x)):
            return "—"
    except (TypeError, ValueError):
        return "—"

    v = abs(float(x))
    signo = "-" if float(x) < 0 else ""
    if v >= 1e12:
        cuerpo = f"{v / 1e12:.2f}".replace(".", ",") + " B"
    elif v >= 1e9:
        cuerpo = f"{v / 1e9:.2f}".replace(".", ",") + " mil M"
    elif v >= 1e6:
        cuerpo = f"{v / 1e6:.0f} M"
    else:
        cuerpo = f"{v:,.0f}".replace(",", ".")
    return f"{signo}${cuerpo}"


def fmt_entero(n) -> str:
    """Separador de miles a la colombiana: 97.512, no 97,512."""
    try:
        return f"{int(n):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def normalizar(texto) -> str:
    """
    Minúsculas y sin tildes, para buscar como escribe la gente.

    Nadie va a teclear «BOLÍVAR» con tilde ni «CORPORACIÓN» con la o acentuada;
    quien busca a una empresa escribe lo que recuerda. Sin esto, la búsqueda
    fallaría justo en los nombres más comunes del dataset.

    Replica paso por paso lo que se aplica a la columna `texto_busqueda` en
    `enriquecer`. Tienen que coincidir exactamente: si esta función conservara
    algún carácter que allí se descarta —y los datos del SECOP traen unos cuantos
    con la codificación dañada— esa búsqueda no encontraría nunca nada.
    """
    if not isinstance(texto, str):
        return ""
    return (
        unicodedata.normalize("NFKD", texto)
        .encode("ascii", errors="ignore")
        .decode("ascii")
        .lower()
    )


def filtrar_por_texto(d: pd.DataFrame, consulta: str) -> pd.DataFrame:
    """
    Filas que contienen **todas** las palabras buscadas, en cualquier orden.

    Buscar por palabras sueltas y no por la frase entera evita que «neurodinamia
    sas» falle solo porque en el dato figure como «NEURODINAMIA S.A.S.».
    """
    palabras = normalizar(consulta or "").split()
    if not palabras or "texto_busqueda" not in d.columns:
        return d

    mascara = pd.Series(True, index=d.index)
    for palabra in palabras:
        mascara &= d["texto_busqueda"].str.contains(palabra, regex=False, na=False)
    return d[mascara]


def abreviar(nombres, n: int = 28) -> list[str]:
    """
    Acorta etiquetas de eje garantizando que sigan siendo distintas.

    Plotly agrupa las categorías por su texto: dos nombres largos que empiezan
    igual se truncarían al mismo valor y sus barras se fundirían en una sola.
    Cuando eso ocurre, se numeran para mantenerlas separadas.
    """
    usadas: set[str] = set()
    salida: list[str] = []
    for nombre in nombres:
        texto = str(nombre)
        corto = texto if len(texto) <= n else texto[: n - 1].rstrip() + "…"
        base, k = corto, 1
        while corto in usadas:
            k += 1
            corto = f"{base} ({k})"
        usadas.add(corto)
        salida.append(corto)
    return salida


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def _extraer_url(valor):
    """
    El SECOP entrega el enlace envuelto en un diccionario (`{'url': '…'}`), pero
    no siempre: conviene aceptar también la cadena suelta y devolver None ante
    cualquier otra cosa, para no colar basura en un enlace que el usuario va a pulsar.
    """
    if isinstance(valor, dict):
        valor = valor.get("url")
    if isinstance(valor, str) and valor.startswith("http"):
        return valor
    return None


def cargar_datos() -> tuple[pd.DataFrame, Fuente]:
    """
    Lee el parquet local si está junto a la app; si no, lo descarga.

    Cualquier fallo se traduce a ErrorDeDatos con un mensaje accionable, para
    que la interfaz pueda explicarlo en vez de mostrar un traceback.
    """
    if os.path.exists(ARCHIVO_LOCAL):
        try:
            with open(ARCHIVO_LOCAL, "rb") as fh:
                crudo = fh.read()
        except OSError as e:
            raise ErrorDeDatos(f"No se pudo leer «{ARCHIVO_LOCAL}»: {e}") from e
        origen = f"archivo local · {ARCHIVO_LOCAL}"
    else:
        try:
            with urllib.request.urlopen(URL_DATOS, timeout=TIMEOUT_DESCARGA) as r:
                crudo = r.read()
        except urllib.error.HTTPError as e:
            raise ErrorDeDatos(
                f"El servidor de datos respondió {e.code} al pedir {URL_DATOS}. "
                "Verifica que la ruta y la referencia (SECOP_DATA_REF) existan."
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ErrorDeDatos(
                f"No se pudo descargar el dataset desde {URL_DATOS} "
                f"(límite {TIMEOUT_DESCARGA}s): {e}. Si el servidor no tiene salida "
                f"a internet, incluye «{ARCHIVO_LOCAL}» junto a la app."
            ) from e
        origen = f"{REPO_DATOS}@{REF_DATOS}"

    try:
        disponibles = set(pq.read_schema(io.BytesIO(crudo)).names)
    except Exception as e:  # pyarrow lanza tipos muy variados
        raise ErrorDeDatos(f"El archivo se descargó pero no es un parquet legible: {e}") from e

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in disponibles]
    if faltantes:
        raise ErrorDeDatos(
            "El dataset cambió de formato: faltan las columnas "
            + ", ".join(f"«{c}»" for c in faltantes)
        )

    # El dataset trae 85 columnas y el observatorio usa una docena. Leyendo solo
    # esas, la memoria del proceso baja alrededor del 90% (de ~195 MB a ~19 MB),
    # que en un servidor modesto es la diferencia entre aguantar una punta de
    # visitas o quedarse sin RAM.
    columnas = list(COLUMNAS_REQUERIDAS) + [c for c in COLUMNAS_OPCIONALES if c in disponibles]
    try:
        df = pd.read_parquet(io.BytesIO(crudo), columns=columnas)
    except Exception as e:
        raise ErrorDeDatos(f"El parquet no se pudo leer: {e}") from e

    fuente = Fuente(
        origen=origen,
        huella=hashlib.sha256(crudo).hexdigest()[:12],
        n_bytes=len(crudo),
    )
    return df, fuente


# --------------------------------------------------------------------------- #
# Enriquecimiento: variables derivadas, señales de riesgo y anomalías
# --------------------------------------------------------------------------- #
def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, sobre TODO el universo de contratos, las columnas de análisis.

    Los umbrales salen de los propios datos (percentiles), así que dependen del
    universo completo y no de lo que el usuario esté filtrando: los filtros de
    la interfaz solo eligen qué contratos mostrar, nunca redefinen qué es atípico.

    Devuelve una copia; el DataFrame de entrada no se modifica.
    """
    d = df.copy()

    # --- Variables derivadas -------------------------------------------------
    valor = pd.to_numeric(d["Valor del Contrato"], errors="coerce")
    d["Valor del Contrato"] = valor
    adiciones = pd.to_numeric(d["Dias adicionados"], errors="coerce")

    d["duracion_dias"] = (
        d["Fecha de Fin del Contrato"] - d["Fecha de Inicio del Contrato"]
    ).dt.days
    # Un contrato que termina antes de empezar es un error de captura de fechas,
    # no una duración negativa real. Se marca y se trata como dato ausente para
    # que no entre al modelo como si fuera una observación válida.
    d["duracion_invalida"] = (d["duracion_dias"] < 0).fillna(False)
    d.loc[d["duracion_invalida"], "duracion_dias"] = np.nan

    d["es_directa"] = (
        d["Modalidad de Contratacion"].astype(str)
        .str.contains("directa", case=False, na=False)
        .astype(int)
    )
    d["Rango Valor"] = pd.cut(valor, bins=_BINS_VALOR, labels=RANGOS_VALOR, right=False)
    d["anio_firma"] = d["Fecha de Firma"].dt.year

    # Enlace a la ficha oficial del proceso en el SECOP. Es lo que permite que
    # quien vea un contrato señalado aquí vaya a comprobarlo en la fuente en vez
    # de tener que creerse este tablero.
    # Un único texto por contrato donde buscar: entidad, proveedor y referencia.
    # Se calcula una vez aquí en vez de normalizar tres columnas en cada
    # búsqueda, que con casi cien mil filas se notaría en cada pulsación.
    d["texto_busqueda"] = (
        d["Nombre Entidad"].astype(str).fillna("")
        + " | "
        + d["Proveedor Adjudicado"].astype(str).fillna("")
        + " | "
        + d["Referencia del Contrato"].astype(str).fillna("")
    ).str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii").str.lower()

    if COLUMNA_URL in d.columns:
        # La columna original guarda un diccionario de Python por fila y cuesta
        # unos 19 MB; una vez extraída la cadena, no aporta nada.
        d["url_secop"] = d[COLUMNA_URL].map(_extraer_url)
        d = d.drop(columns=[COLUMNA_URL])
    else:
        d["url_secop"] = None

    # --- Señal 1: valor inusualmente alto ------------------------------------
    umbral_valor = valor.quantile(PCT_VALOR_ALTO)
    d["flag_valor"] = (valor > umbral_valor).fillna(False)

    # --- Señal 2: la entidad recurre a contratación directa más que sus pares -
    # La contratación directa es la norma en Cartagena (en torno al 84% de los
    # contratos), de modo que marcarla por sí sola no distingue nada: dejaría a
    # cinco de cada seis contratos "señalados". Lo informativo no es usarla, sino
    # usarla por encima de lo que hacen entidades comparables. Solo se juzga a
    # entidades con volumen suficiente para que su tasa sea estable.
    por_entidad = (
        d.groupby("Nombre Entidad", observed=True)["es_directa"]
        .agg(tasa="mean", n="size")
    )
    evaluables = por_entidad[por_entidad["n"] >= MIN_CONTRATOS_ENTIDAD]
    umbral_directa = (
        float(evaluables["tasa"].quantile(PCT_DIRECTA_PARES)) if len(evaluables) else np.nan
    )
    d["tasa_directa_entidad"] = d["Nombre Entidad"].map(por_entidad["tasa"])
    d["contratos_entidad"] = d["Nombre Entidad"].map(por_entidad["n"])
    if pd.notna(umbral_directa):
        d["flag_directa"] = (
            (d["es_directa"] == 1)
            & (d["contratos_entidad"] >= MIN_CONTRATOS_ENTIDAD)
            & (d["tasa_directa_entidad"] > umbral_directa)
        )
    else:
        d["flag_directa"] = False

    # --- Señal 3: adiciones de plazo extensas --------------------------------
    adiciones_positivas = adiciones[adiciones > 0]
    umbral_dias = (
        float(adiciones_positivas.quantile(PCT_ADICIONES))
        if len(adiciones_positivas) else np.nan
    )
    d["flag_adiciones"] = (
        (adiciones.fillna(0) > umbral_dias) if pd.notna(umbral_dias) else False
    )

    d["senales"] = (
        d[["flag_valor", "flag_directa", "flag_adiciones"]].astype(bool).sum(axis=1).astype(int)
    )
    d["Riesgo"] = pd.Categorical(
        d["senales"].map(NIVELES_RIESGO),
        categories=list(NIVELES_RIESGO.values()),
        ordered=True,
    )
    d["reglas_riesgo"] = d["senales"] >= 2

    # --- Isolation Forest ----------------------------------------------------
    # El valor del contrato abarca varios órdenes de magnitud y está muy sesgado
    # a la derecha. Sin transformar, domina la distancia y el modelo se limita a
    # redescubrir "los contratos más caros", que es justo lo que la regla del p95
    # ya detecta. log1p lo lleva a una escala comparable con la duración y las
    # adiciones, de modo que el modelo aporte combinaciones inusuales y no un
    # ranking de precios.
    X = pd.DataFrame(
        {
            "log_valor": np.log1p(valor.clip(lower=0)).astype(float),
            "duracion_dias": pd.to_numeric(d["duracion_dias"], errors="coerce").astype(float),
            "dias_adicionados": adiciones.fillna(0).astype(float),
            "es_directa": d["es_directa"].astype(float),
        },
        index=d.index,
    )
    X["duracion_dias"] = X["duracion_dias"].fillna(X["duracion_dias"].median())
    usable = X.notna().all(axis=1)

    d["es_anomalo"] = False
    d["anomaly_score"] = np.nan
    if int(usable.sum()) >= MIN_FILAS_MODELO:
        X_esc = StandardScaler().fit_transform(X[usable])
        iso = IsolationForest(
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS,
        )
        d.loc[usable, "es_anomalo"] = iso.fit_predict(X_esc) == -1
        d.loc[usable, "anomaly_score"] = iso.decision_function(X_esc)

    d.attrs.update(
        {
            "umbral_valor": umbral_valor,
            "umbral_dias": umbral_dias,
            "umbral_directa": umbral_directa,
            "entidades_evaluables": int(len(evaluables)),
            "entidades_totales": int(len(por_entidad)),
            "filas_modelo": int(usable.sum()),
            "n_duracion_invalida": int(d["duracion_invalida"].sum()),
            "n_sin_duracion": int(d["duracion_dias"].isna().sum()),
            "n_sin_valor": int(valor.isna().sum()),
        }
    )
    return d


def rango_anios(d: pd.DataFrame) -> tuple[int, int]:
    """Primer y último año de firma presentes, para no fijar la ventana a mano."""
    anios = d["anio_firma"].dropna()
    if anios.empty:
        return (0, 0)
    return int(anios.min()), int(anios.max())


def resumen_calidad(d: pd.DataFrame) -> dict:
    """Cifras de calidad del dato que la interfaz muestra de forma explícita."""
    return {
        "sin_duracion": int(d.attrs.get("n_sin_duracion", 0)),
        "fechas_invertidas": int(d.attrs.get("n_duracion_invalida", 0)),
        "sin_valor": int(d.attrs.get("n_sin_valor", 0)),
        "filas_modelo": int(d.attrs.get("filas_modelo", 0)),
        "total": int(len(d)),
    }
