# -*- coding: utf-8 -*-
"""
Pruebas de la interfaz.

`AppTest` ejecuta app.py de verdad —el mismo camino que recorre el navegador—
y recoge las excepciones. Cubre lo que las pruebas del núcleo no ven: llamadas a
Streamlit con parámetros que la versión instalada no acepta, gráficas que fallan
con una vista vacía, o textos que interpolan una columna inexistente.
"""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import pipeline as pl  # noqa: E402
from tests.test_pipeline import construir_df, universo  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


@pytest.fixture(autouse=True)
def _cache_limpia():
    """
    `st.cache_data` es global al proceso y `cargar()` no recibe argumentos, así
    que sin limpiarla cada prueba heredaría el dataset de la anterior y pasaría
    verificando datos que no son los suyos.
    """
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


def arrancar(tmp_path, monkeypatch, df):
    """Deja un dataset local en el directorio de trabajo y ejecuta la app."""
    df.to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def sin_excepciones(at):
    return [f"{e.value}" for e in at.exception]


def _valores_url(at, clave):
    """
    Los parámetros de la URL, siempre como lista.

    AppTest los expone en un diccionario donde el valor puede venir suelto o en
    lista, según cómo se haya escrito; el navegador siempre da la cadena.
    """
    valor = at.query_params.get(clave, [])
    return valor if isinstance(valor, list) else [valor]


def test_la_app_arranca_sin_excepciones(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    assert sin_excepciones(at) == []


def test_la_app_muestra_los_seis_kpis(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    # Los KPIs se pintan como HTML dentro de columnas; basta con comprobar que
    # sus etiquetas llegaron a la página.
    html = " ".join(m.value for m in at.markdown)
    for etiqueta in ("Contratos", "Valor total", "Valor promedio",
                     "Duración media", "Participación Pyme", "Contratos atípicos"):
        assert etiqueta in html


def test_la_app_declara_la_version_del_dataset(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    textos = " ".join(c.value for c in at.caption)
    assert "versión" in textos and pl.ARCHIVO_LOCAL in textos


def test_la_app_sobrevive_a_una_vista_sin_contratos(tmp_path, monkeypatch):
    """El filtro de 'solo riesgo' puede dejar la vista vacía: no debe romperse."""
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.checkbox[0].check().run()
    assert sin_excepciones(at) == []


def test_filtrar_por_entidad_no_rompe_las_graficas(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.multiselect[1].select("CHICA").run()
    assert sin_excepciones(at) == []


def test_la_app_explica_el_fallo_en_vez_de_reventar(tmp_path, monkeypatch):
    """Sin datos locales y sin red, el usuario debe ver un error legible."""
    import urllib.error

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pl.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("sin red")),
    )
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()

    assert sin_excepciones(at) == []
    assert any("No se pudieron cargar los datos" in e.value for e in at.error)


def test_un_dataset_pequeno_no_rompe_la_app(tmp_path, monkeypatch):
    """Por debajo del mínimo del modelo la app debe seguir siendo utilizable."""
    filas = [{"entidad": "E", "modalidad": "Licitación", "valor": 1e6} for _ in range(5)]
    at = arrancar(tmp_path, monkeypatch, construir_df(filas))
    assert sin_excepciones(at) == []


def test_un_solo_proveedor_no_genera_una_categoria_otros_falsa(tmp_path, monkeypatch):
    """Con 6 proveedores o menos no debe aparecer una barra 'Otros' vacía."""
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "proveedor": f"P{i % 3}"}
        for i in range(60)
    ]
    d = pl.enriquecer(construir_df(filas))
    prov = d.groupby("Proveedor Adjudicado", observed=True)["Valor del Contrato"].sum()
    assert len(prov) == 3
    at = arrancar(tmp_path, monkeypatch, construir_df(filas))
    assert sin_excepciones(at) == []


def test_las_fechas_faltantes_no_rompen_la_app(tmp_path, monkeypatch):
    df = universo()
    df.loc[df.index[:20], "Fecha de Firma"] = pd.NaT
    df.loc[df.index[:20], "Fecha de Inicio del Contrato"] = pd.NaT
    at = arrancar(tmp_path, monkeypatch, df)
    assert sin_excepciones(at) == []


# --------------------------------------------------------------------------- #
# Enlaces compartibles: los filtros viajan en la dirección de la página
# --------------------------------------------------------------------------- #
def test_los_filtros_quedan_reflejados_en_la_url(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.checkbox[0].check().run()
    at.sidebar.multiselect[0].select("Licitación pública").run()

    assert "1" in _valores_url(at, "riesgo")
    assert "Licitación pública" in _valores_url(at, "modalidad")
    assert sin_excepciones(at) == []


def test_una_url_compartida_reconstruye_la_misma_vista(tmp_path, monkeypatch):
    universo().to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["modalidad"] = "Licitación pública"
    at.query_params["riesgo"] = "1"
    at.run()

    assert sin_excepciones(at) == []
    assert at.sidebar.checkbox[0].value is True
    assert at.sidebar.multiselect[0].value == ["Licitación pública"]


def test_una_vista_sin_filtros_no_ensucia_la_url(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    assert dict(at.query_params) == {}


def test_un_enlace_viejo_o_manipulado_no_rompe_la_app(tmp_path, monkeypatch):
    """
    Los enlaces se comparten y sobreviven a los cambios del dataset: años fuera
    de rango, entidades que ya no existen o basura escrita a mano no pueden
    dejar la página en blanco.
    """
    universo().to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["desde"] = "1900"
    at.query_params["hasta"] = "no es un año"
    at.query_params["entidad"] = "ENTIDAD QUE NO EXISTE"
    at.query_params["modalidad"] = "<script>alert(1)</script>"
    at.run()

    assert sin_excepciones(at) == []
    assert at.sidebar.multiselect[1].value == []


# --------------------------------------------------------------------------- #
# Enlace al contrato original
# --------------------------------------------------------------------------- #
def test_las_tablas_ofrecen_el_enlace_al_contrato_original(tmp_path, monkeypatch):
    df = universo()
    df[pl.COLUMNA_URL] = [{"url": f"https://community.secop.gov.co/{i}"} for i in range(len(df))]
    at = arrancar(tmp_path, monkeypatch, df)

    assert sin_excepciones(at) == []
    enlaces = [
        c
        for d in at.dataframe
        for c in d.value.columns
        if c == "url_secop"
    ]
    assert enlaces, "ninguna tabla incluye el enlace al SECOP"


# --------------------------------------------------------------------------- #
# Buscador
# --------------------------------------------------------------------------- #
def test_buscar_acota_la_vista(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.text_input[0].set_value("chica").run()

    assert sin_excepciones(at) == []
    assert "5" in [m.value for m in at.sidebar.metric]


def test_la_busqueda_viaja_en_la_url(tmp_path, monkeypatch):
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.text_input[0].set_value("alta").run()
    assert "alta" in _valores_url(at, "q")


def test_una_url_con_busqueda_la_restaura(tmp_path, monkeypatch):
    universo().to_parquet(tmp_path / pl.ARCHIVO_LOCAL)
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["q"] = "chica"
    at.run()

    assert sin_excepciones(at) == []
    assert at.sidebar.text_input[0].value == "chica"


def test_una_busqueda_sin_resultados_lo_explica(tmp_path, monkeypatch):
    """Una pantalla vacía sin explicación parece que la app se rompió."""
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.text_input[0].set_value("zzzz-no-existe").run()

    assert sin_excepciones(at) == []
    assert any("Nada coincide" in w.value for w in at.sidebar.warning)


def test_buscar_sin_tildes_encuentra_lo_acentuado(tmp_path, monkeypatch):
    df = universo()
    df["Nombre Entidad"] = df["Nombre Entidad"].replace({"ALTA": "CORPORACIÓN AUTÓNOMA"})
    at = arrancar(tmp_path, monkeypatch, df)
    at.sidebar.text_input[0].set_value("corporacion autonoma").run()

    assert sin_excepciones(at) == []
    assert "40" in [m.value for m in at.sidebar.metric]


def test_al_quitar_los_filtros_la_url_vuelve_a_quedar_limpia(tmp_path, monkeypatch):
    """
    Si la dirección conserva un filtro ya borrado, quien copie el enlace comparte
    una vista que el remitente no está viendo.
    """
    at = arrancar(tmp_path, monkeypatch, universo())
    at.sidebar.text_input[0].set_value("alta").run()
    assert "alta" in _valores_url(at, "q")

    at.sidebar.text_input[0].set_value("").run()
    assert dict(at.query_params) == {}
    assert sin_excepciones(at) == []
