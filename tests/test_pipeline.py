# -*- coding: utf-8 -*-
"""
Pruebas del núcleo analítico.

Todo se comprueba sobre datos sintéticos construidos a mano: así los umbrales
esperados se pueden calcular de cabeza y las pruebas no dependen de que el
dataset real esté descargado ni de que no haya cambiado esta semana.
"""
import numpy as np
import pandas as pd
import pytest

import pipeline as pl
from pipeline import ErrorDeDatos


# --------------------------------------------------------------------------- #
# Datos de prueba
# --------------------------------------------------------------------------- #
def construir_df(filas):
    """Arma un DataFrame con la forma que espera `enriquecer`."""
    base = pd.Timestamp("2023-01-01")
    return pd.DataFrame(
        {
            "Nombre Entidad": [r["entidad"] for r in filas],
            "ID Contrato": [f"C{i:05d}" for i in range(len(filas))],
            "Referencia del Contrato": [f"REF-{i:05d}" for i in range(len(filas))],
            "Modalidad de Contratacion": [r["modalidad"] for r in filas],
            "Proveedor Adjudicado": [r.get("proveedor", "PROV") for r in filas],
            "Es Pyme": [r.get("pyme", True) for r in filas],
            "Valor del Contrato": [float(r["valor"]) for r in filas],
            "Dias adicionados": pd.array([r.get("adiciones", 0) for r in filas], dtype="Int64"),
            "Fecha de Firma": [base + pd.Timedelta(days=r.get("firma", 0)) for r in filas],
            "Fecha de Inicio del Contrato": [
                base + pd.Timedelta(days=r.get("inicio", 0)) for r in filas
            ],
            "Fecha de Fin del Contrato": [
                base + pd.Timedelta(days=r.get("fin", 100)) for r in filas
            ],
        }
    )


def universo():
    """
    Cuatro entidades con tasas de contratación directa conocidas:

        ALTA   40 contratos, 100% directa   -> tasa 1.00
        MEDIA  40 contratos,  50% directa   -> tasa 0.50
        BAJA   40 contratos,   0% directa   -> tasa 0.00
        CHICA   5 contratos, 100% directa   -> tasa 1.00 pero sin volumen

    El percentil 75 de {1.00, 0.50, 0.00} es 0.75, así que solo ALTA lo supera.
    CHICA queda fuera por no llegar a MIN_CONTRATOS_ENTIDAD.
    """
    filas = []
    for i in range(40):
        filas.append({"entidad": "ALTA", "modalidad": "Contratación directa", "valor": 1e6 + i})
    for i in range(40):
        modalidad = "Contratación directa" if i < 20 else "Licitación pública"
        filas.append({"entidad": "MEDIA", "modalidad": modalidad, "valor": 2e6 + i})
    for i in range(40):
        filas.append({"entidad": "BAJA", "modalidad": "Licitación pública", "valor": 3e6 + i})
    for i in range(5):
        filas.append({"entidad": "CHICA", "modalidad": "Contratación directa", "valor": 4e6 + i})
    return construir_df(filas)


@pytest.fixture(scope="module")
def enriquecido():
    return pl.enriquecer(universo())


# --------------------------------------------------------------------------- #
# fmt_cop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "valor, esperado",
    [
        (0, "$0"),
        (1234, "$1.234"),
        (5_000_000, "$5 M"),
        (1_889_133_421, "$1,89 mil M"),
        (2.5e12, "$2,50 B"),
        (-5_000_000, "-$5 M"),
    ],
)
def test_fmt_cop_usa_convencion_colombiana(valor, esperado):
    assert pl.fmt_cop(valor) == esperado


@pytest.mark.parametrize("vacio", [None, np.nan, pd.NA, float("inf"), "no es un número"])
def test_fmt_cop_tolera_valores_no_representables(vacio):
    assert pl.fmt_cop(vacio) == "—"


# --------------------------------------------------------------------------- #
# abreviar
# --------------------------------------------------------------------------- #
def test_abreviar_respeta_nombres_cortos():
    assert pl.abreviar(["corto", "otro"], 20) == ["corto", "otro"]


def test_abreviar_desambigua_nombres_que_truncarian_igual():
    # Sin desambiguación, Plotly fundiría estas dos barras en una sola.
    salida = pl.abreviar(["CONSORCIO NACIONAL UNO", "CONSORCIO NACIONAL DOS"], 15)
    assert len(set(salida)) == 2, salida


def test_abreviar_no_excede_el_largo_pedido_sin_colisiones():
    assert all(len(s) <= 15 for s in pl.abreviar(["A" * 40, "B" * 40], 15))


# --------------------------------------------------------------------------- #
# enriquecer: contrato de la función
# --------------------------------------------------------------------------- #
def test_enriquecer_no_modifica_el_dataframe_de_entrada():
    df = universo()
    columnas_antes = list(df.columns)
    pl.enriquecer(df)
    assert list(df.columns) == columnas_antes


def test_enriquecer_expone_los_umbrales_usados(enriquecido):
    for clave in ("umbral_valor", "umbral_dias", "umbral_directa", "entidades_evaluables"):
        assert clave in enriquecido.attrs


# --------------------------------------------------------------------------- #
# Señal 1: valor
# --------------------------------------------------------------------------- #
def test_flag_valor_marca_exactamente_por_encima_del_p95(enriquecido):
    umbral = enriquecido.attrs["umbral_valor"]
    esperado = enriquecido["Valor del Contrato"] > umbral
    pd.testing.assert_series_equal(
        enriquecido["flag_valor"], esperado, check_names=False
    )


# --------------------------------------------------------------------------- #
# Señal 2: contratación directa relativa a los pares
# --------------------------------------------------------------------------- #
def test_flag_directa_marca_solo_a_la_entidad_intensiva(enriquecido):
    marcadas = set(enriquecido.loc[enriquecido["flag_directa"], "Nombre Entidad"].unique())
    assert marcadas == {"ALTA"}


def test_flag_directa_ignora_entidades_sin_volumen_suficiente(enriquecido):
    # CHICA también contrata 100% directa, pero con 5 contratos su tasa no es
    # evidencia de nada: no debe señalarse.
    chica = enriquecido[enriquecido["Nombre Entidad"] == "CHICA"]
    assert not chica["flag_directa"].any()


def test_flag_directa_no_marca_contratos_no_directos(enriquecido):
    assert not (enriquecido["flag_directa"] & (enriquecido["es_directa"] == 0)).any()


def test_la_senal_de_directa_es_discriminante(enriquecido):
    # El punto de medirla contra los pares: si marcara la mayoría del universo
    # (como hacía marcar "es directa" a secas) no aportaría información.
    assert enriquecido["flag_directa"].mean() < 0.5
    assert enriquecido["es_directa"].mean() > 0.5  # directa sí es mayoritaria


def test_umbral_directa_es_el_percentil_75_de_las_entidades_evaluables(enriquecido):
    assert enriquecido.attrs["umbral_directa"] == pytest.approx(0.75)
    assert enriquecido.attrs["entidades_evaluables"] == 3
    assert enriquecido.attrs["entidades_totales"] == 4


# --------------------------------------------------------------------------- #
# Señal 3: adiciones
# --------------------------------------------------------------------------- #
def test_flag_adiciones_usa_el_p90_de_las_adiciones_positivas():
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "adiciones": a}
        for a in list(range(1, 101)) + [0] * 50
    ]
    d = pl.enriquecer(construir_df(filas))
    # p90 de 1..100 es 90.1; solo los estrictamente mayores se marcan.
    assert d.attrs["umbral_dias"] == pytest.approx(90.1)
    assert d["flag_adiciones"].sum() == 10


def test_sin_adiciones_positivas_la_senal_queda_apagada():
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "adiciones": 0}
        for _ in range(60)
    ]
    d = pl.enriquecer(construir_df(filas))
    assert not d["flag_adiciones"].any()
    assert pd.isna(d.attrs["umbral_dias"])


# --------------------------------------------------------------------------- #
# Agregación de señales
# --------------------------------------------------------------------------- #
def test_senales_y_riesgo_son_coherentes(enriquecido):
    esperado = (
        enriquecido[["flag_valor", "flag_directa", "flag_adiciones"]].astype(bool).sum(axis=1)
    )
    assert (enriquecido["senales"] == esperado).all()
    assert (enriquecido["senales"].between(0, 3)).all()
    assert (
        enriquecido["Riesgo"].astype(str)
        == enriquecido["senales"].map(pl.NIVELES_RIESGO)
    ).all()
    assert (enriquecido["reglas_riesgo"] == (enriquecido["senales"] >= 2)).all()


# --------------------------------------------------------------------------- #
# Calidad del dato
# --------------------------------------------------------------------------- #
def test_las_fechas_invertidas_no_producen_duraciones_negativas():
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "inicio": 10, "fin": 0}
    ] + [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "inicio": 0, "fin": 30}
        for _ in range(59)
    ]
    d = pl.enriquecer(construir_df(filas))
    assert d.attrs["n_duracion_invalida"] == 1
    assert not (d["duracion_dias"] < 0).any()
    assert d["duracion_dias"].isna().sum() == 1


def test_resumen_calidad_reporta_los_totales(enriquecido):
    cal = pl.resumen_calidad(enriquecido)
    assert cal["total"] == len(enriquecido)
    assert cal["fechas_invertidas"] == 0
    assert set(cal) == {"sin_duracion", "fechas_invertidas", "sin_valor", "filas_modelo", "total"}


# --------------------------------------------------------------------------- #
# Isolation Forest
# --------------------------------------------------------------------------- #
def test_el_modelo_marca_aproximadamente_la_fraccion_configurada():
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6 + i, "fin": 100 + i}
        for i in range(1000)
    ]
    d = pl.enriquecer(construir_df(filas))
    assert d["es_anomalo"].mean() == pytest.approx(pl.CONTAMINATION, abs=0.01)
    assert d.loc[d["es_anomalo"], "anomaly_score"].notna().all()


def test_el_modelo_es_reproducible():
    a = pl.enriquecer(universo())["es_anomalo"]
    b = pl.enriquecer(universo())["es_anomalo"]
    pd.testing.assert_series_equal(a, b)


def test_con_muy_pocas_filas_el_modelo_se_omite_sin_fallar():
    filas = [{"entidad": "E", "modalidad": "Licitación", "valor": 1e6} for _ in range(10)]
    d = pl.enriquecer(construir_df(filas))
    assert not d["es_anomalo"].any()
    assert d["anomaly_score"].isna().all()


def test_las_anomalias_no_son_solo_los_contratos_mas_caros():
    """
    El valor entra en escala logarítmica justamente para esto: si entrara crudo,
    dominaría la distancia y el modelo se limitaría a rankear precios, duplicando
    lo que ya hace la regla del p95.

    Se construyen 200 contratos cuyo valor cubre seis órdenes de magnitud y se
    colocan dos duraciones desmedidas en contratos *baratos*. Un detector guiado
    por el precio los pasaría por alto.
    """
    valores = np.geomspace(1e6, 1e12, 200)
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": v, "fin": 100}
        for v in valores
    ]
    for i in (5, 6):
        filas[i]["fin"] = 5000  # contrato barato, duración desmedida

    d = pl.enriquecer(construir_df(filas))

    assert d["es_anomalo"].iloc[[5, 6]].all(), "no detectó las duraciones desmedidas"
    baratos = d["Valor del Contrato"] < d["Valor del Contrato"].median()
    assert (d["es_anomalo"] & baratos).any(), "todas las anomalías son contratos caros"


# --------------------------------------------------------------------------- #
# Rango de años
# --------------------------------------------------------------------------- #
def test_rango_anios_se_deriva_de_los_datos():
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "firma": d}
        for d in (0, 400, 800)
    ]
    d = pl.enriquecer(construir_df(filas))
    assert pl.rango_anios(d) == (2023, 2025)


def test_rango_anios_sin_fechas_no_falla():
    df = construir_df([{"entidad": "E", "modalidad": "L", "valor": 1e6}])
    df["Fecha de Firma"] = pd.NaT
    assert pl.rango_anios(pl.enriquecer(df)) == (0, 0)


# --------------------------------------------------------------------------- #
# Carga de datos: los fallos se traducen a un error explicable
# --------------------------------------------------------------------------- #
def test_un_dataset_sin_las_columnas_esperadas_da_error_claro(tmp_path, monkeypatch):
    incompleto = tmp_path / pl.ARCHIVO_LOCAL
    pd.DataFrame({"otra_cosa": [1, 2, 3]}).to_parquet(incompleto)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ErrorDeDatos) as exc:
        pl.cargar_datos()
    assert "faltan las columnas" in str(exc.value)


def test_un_fallo_de_red_da_error_explicable(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.chdir(tmp_path)  # sin archivo local -> intenta descargar

    def falla(*args, **kwargs):
        raise urllib.error.URLError("sin ruta al host")

    monkeypatch.setattr(pl.urllib.request, "urlopen", falla)

    with pytest.raises(ErrorDeDatos) as exc:
        pl.cargar_datos()
    assert "No se pudo descargar" in str(exc.value)


def test_la_carga_local_devuelve_una_huella_del_contenido(tmp_path, monkeypatch):
    universo().to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)

    df, fuente = pl.cargar_datos()
    assert len(df) == 125
    assert len(fuente.huella) == 12
    assert fuente.n_bytes > 0
    assert pl.ARCHIVO_LOCAL in fuente.origen


# --------------------------------------------------------------------------- #
# Enlace al contrato original en el SECOP
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "crudo, esperado",
    [
        ({"url": "https://community.secop.gov.co/x"}, "https://community.secop.gov.co/x"),
        ("https://community.secop.gov.co/y", "https://community.secop.gov.co/y"),
        ({"url": None}, None),
        ({}, None),
        ("javascript:alert(1)", None),
        ("no es una url", None),
        (None, None),
        (123, None),
    ],
)
def test_solo_se_aceptan_enlaces_con_forma_de_url(crudo, esperado):
    # El valor acaba siendo un enlace que alguien va a pulsar: nada que no
    # empiece por http debe llegar hasta ahí.
    assert pl._extraer_url(crudo) == esperado


def test_enriquecer_extrae_el_enlace_y_descarta_la_columna_cruda():
    df = universo()
    df[pl.COLUMNA_URL] = [{"url": f"https://secop.gov.co/{i}"} for i in range(len(df))]
    d = pl.enriquecer(df)
    assert d["url_secop"].iloc[0] == "https://secop.gov.co/0"
    assert pl.COLUMNA_URL not in d.columns


def test_la_app_funciona_si_el_dataset_no_trae_enlaces():
    d = pl.enriquecer(universo())
    assert "url_secop" in d.columns
    assert d["url_secop"].isna().all()


# --------------------------------------------------------------------------- #
# Solo se leen las columnas que se usan
# --------------------------------------------------------------------------- #
def test_la_carga_ignora_las_columnas_que_no_se_usan(tmp_path, monkeypatch):
    """
    El dataset real trae 85 columnas y el observatorio usa una docena. Leerlas
    todas multiplicaba por cinco la memoria del proceso.
    """
    df = universo()
    for i in range(20):
        df[f"columna_inutil_{i}"] = "x" * 50
    df.to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)

    leido, _ = pl.cargar_datos()
    assert not [c for c in leido.columns if c.startswith("columna_inutil")]
    assert set(pl.COLUMNAS_REQUERIDAS) <= set(leido.columns)


def test_la_carga_toma_el_enlace_cuando_el_dataset_lo_trae(tmp_path, monkeypatch):
    df = universo()
    df[pl.COLUMNA_URL] = [{"url": "https://secop.gov.co/z"}] * len(df)
    df.to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)

    leido, _ = pl.cargar_datos()
    assert pl.COLUMNA_URL in leido.columns


# --------------------------------------------------------------------------- #
# Búsqueda libre
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ("BOLÍVAR", "bolivar"),
        ("Corporación Autónoma", "corporacion autonoma"),
        ("NEURODINAMIA S.A.S.", "neurodinamia s.a.s."),
        ("Niño", "nino"),
        (None, ""),
        (123, ""),
    ],
)
def test_normalizar_quita_tildes_y_baja_a_minusculas(entrada, esperado):
    assert pl.normalizar(entrada) == esperado


def test_normalizar_coincide_con_la_columna_de_busqueda():
    """
    Si el normalizador de la consulta y el de la columna divergieran, algunas
    búsquedas no encontrarían nada nunca y nadie sabría por qué. Los datos del
    SECOP traen caracteres con la codificación dañada, así que el caso es real.
    """
    raros = [
        "Rama Judicial \x96 Dirección Seccional",   # guion mal codificado, como en el dataset
        "CORPORACIÓN AUTÓNOMA",
        "Niño & Cía. Ltda.",
        "ESE HOSPITAL — CARTAGENA",
    ]
    filas = [{"entidad": r, "modalidad": "L", "valor": 1e6} for r in raros] * 15
    d = pl.enriquecer(construir_df(filas))
    for i, original in enumerate(raros):
        assert pl.normalizar(original) in d["texto_busqueda"].iloc[i], original


def test_la_busqueda_encuentra_por_entidad_y_por_proveedor():
    filas = [
        {"entidad": "ALCALDÍA DE CARTAGENA", "modalidad": "L", "valor": 1e6,
         "proveedor": "CONSTRUCTORA DEL CARIBE"},
    ] + [{"entidad": "OTRA", "modalidad": "L", "valor": 1e6, "proveedor": "OTRO"}] * 59
    d = pl.enriquecer(construir_df(filas))
    assert len(pl.filtrar_por_texto(d, "alcaldia")) == 1
    assert len(pl.filtrar_por_texto(d, "constructora")) == 1
    assert len(pl.filtrar_por_texto(d, "CARIBE")) == 1


def test_la_busqueda_ignora_el_orden_de_las_palabras():
    filas = [
        {"entidad": "HOSPITAL NAVAL DE CARTAGENA", "modalidad": "L", "valor": 1e6}
    ] + [{"entidad": "OTRA", "modalidad": "L", "valor": 1e6}] * 59
    d = pl.enriquecer(construir_df(filas))
    assert len(pl.filtrar_por_texto(d, "hospital naval")) == 1
    assert len(pl.filtrar_por_texto(d, "naval hospital")) == 1
    # Todas las palabras deben aparecer, no basta con una.
    assert len(pl.filtrar_por_texto(d, "hospital aeropuerto")) == 0


def test_una_busqueda_vacia_no_filtra_nada():
    d = pl.enriquecer(universo())
    for vacia in ("", "   ", None):
        assert len(pl.filtrar_por_texto(d, vacia)) == len(d)


def test_la_busqueda_no_interpreta_expresiones_regulares():
    """Un punto o un paréntesis son literales, no comodines."""
    filas = [
        {"entidad": "S.A.S.", "modalidad": "L", "valor": 1e6},
        {"entidad": "SXAXSX", "modalidad": "L", "valor": 1e6},
    ] + [{"entidad": "OTRA", "modalidad": "L", "valor": 1e6}] * 58
    d = pl.enriquecer(construir_df(filas))
    assert len(pl.filtrar_por_texto(d, "s.a.s.")) == 1
    assert len(pl.filtrar_por_texto(d, "(")) == 0


def test_fmt_entero_usa_punto_de_miles():
    assert pl.fmt_entero(97512) == "97.512"
    assert pl.fmt_entero(105.0) == "105"
    assert pl.fmt_entero(0) == "0"
    assert pl.fmt_entero("no es un número") == "—"
