# -*- coding: utf-8 -*-
"""
Pruebas de las lecturas en lenguaje llano.

Un texto explicativo equivocado es peor que no tener ninguno: el ciudadano no
tiene forma de detectar el error, porque justamente vino aquí a que se lo
explicaran. Estas pruebas comprueban que las cifras citadas salen de los datos y
que las frases se sostienen sea cual sea la forma que tomen esos datos.
"""
import pandas as pd
import pytest

import lecturas as lc
import pipeline as pl
from tests.test_pipeline import construir_df, universo

TODAS = [
    lc.lectura_kpis,
    lc.lectura_entidades,
    lc.lectura_modalidades,
    lc.lectura_evolucion,
    lc.lectura_valor_mediano,
    lc.lectura_pyme,
    lc.lectura_pareto,
    lc.lectura_reglas,
    lc.lectura_anomalias,
    lc.lectura_coincidencia,
]


@pytest.fixture(scope="module")
def d():
    return pl.enriquecer(universo())


# --------------------------------------------------------------------------- #
# Robustez
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", TODAS, ids=lambda f: f.__name__)
def test_toda_lectura_devuelve_las_tres_partes(fn, d):
    lec = fn(d)
    assert isinstance(lec, lc.Lectura)
    assert lec.que_es.strip()
    assert lec.que_dice.strip()


@pytest.mark.parametrize("fn", TODAS, ids=lambda f: f.__name__)
def test_ninguna_lectura_falla_con_la_vista_vacia(fn, d):
    lec = fn(d.iloc[0:0])
    assert lec.que_es.strip()
    assert lec.que_dice.strip()


@pytest.mark.parametrize("fn", TODAS, ids=lambda f: f.__name__)
def test_ninguna_lectura_deja_marcadores_sin_rellenar(fn, d):
    texto = " ".join([fn(d).que_es, fn(d).que_dice, fn(d).por_que_importa])
    assert "{" not in texto and "}" not in texto
    assert "nan" not in texto.lower().replace("financ", "")


def test_las_lecturas_de_un_solo_contrato_no_fallan():
    filas = [{"entidad": "E", "modalidad": "Licitación", "valor": 1e6}]
    d = pl.enriquecer(construir_df(filas))
    for fn in TODAS:
        assert fn(d).que_dice.strip()


# --------------------------------------------------------------------------- #
# Formato de las cifras
# --------------------------------------------------------------------------- #
def test_los_miles_usan_punto_y_no_coma():
    """En Colombia 97.512 son noventa y siete mil, no noventa y siete coma cinco."""
    filas = [
        {"entidad": "E", "modalidad": "Licitación", "valor": 1e6, "proveedor": f"P{i}"}
        for i in range(1500)
    ]
    d = pl.enriquecer(construir_df(filas))
    texto = " ".join(lc.resumen_vista(d))
    assert "1.500" in texto
    assert "1,500" not in texto


def test_una_cantidad_pequena_no_se_muestra_como_cero_por_ciento():
    """Decir "0%" de algo que existe haría creer al lector que no hay ninguno."""
    assert lc._pct(3, 1000) == "menos del 1%"
    assert lc._pct(0, 1000) == "0%"
    assert lc._pct(999, 1000) == "100%"
    assert lc._pct(5, 0) == "—"


# --------------------------------------------------------------------------- #
# Frases que dependen de la forma de los datos
# --------------------------------------------------------------------------- #
def test_la_evolucion_dice_mas_cuando_sube_y_menos_cuando_baja():
    def texto_para(valor_ultimo_anio):
        filas = [
            {"entidad": "E", "modalidad": "L", "valor": 1e6, "firma": 0} for _ in range(30)
        ] + [
            {"entidad": "E", "modalidad": "L", "valor": valor_ultimo_anio, "firma": 400}
            for _ in range(30)
        ]
        return lc.lectura_evolucion(pl.enriquecer(construir_df(filas))).que_dice

    subida = texto_para(5e6)
    bajada = texto_para(1e5)
    assert "más** que el año anterior" in subida and "menos**" not in subida
    assert "menos** que el año anterior" in bajada and "más**" not in bajada


def test_la_lectura_de_pyme_no_supone_que_la_participacion_caiga():
    """
    En los datos reales las Pyme participan MÁS en los contratos medianos que en
    los pequeños. Un texto que diera por hecho lo contrario mentiría.
    """
    filas = []
    for i in range(40):  # contratos pequeños, pocas Pyme
        filas.append({"entidad": "E", "modalidad": "L", "valor": 1e6, "pyme": i < 4})
    for i in range(40):  # contratos medianos, muchas Pyme
        filas.append({"entidad": "E", "modalidad": "L", "valor": 2e8, "pyme": i < 36})

    dice = lc.lectura_pyme(pl.enriquecer(construir_df(filas))).que_dice
    assert "mayor en los contratos de **90–500 M**" in dice
    assert "menor en los de **< 90 M**" in dice


def test_el_aviso_de_promedio_inflado_solo_aparece_cuando_lo_esta():
    parejos = [{"entidad": "E", "modalidad": "L", "valor": 1e6} for _ in range(60)]
    dice = lc.lectura_kpis(pl.enriquecer(construir_df(parejos))).que_dice
    assert "inflando el promedio" not in dice

    disparejos = parejos + [{"entidad": "E", "modalidad": "L", "valor": 1e12}]
    dice = lc.lectura_kpis(pl.enriquecer(construir_df(disparejos))).que_dice
    assert "inflando el promedio" in dice


# --------------------------------------------------------------------------- #
# Las cifras citadas salen de los datos
# --------------------------------------------------------------------------- #
def test_el_pareto_cita_el_numero_real_de_proveedores():
    filas = [
        {"entidad": "E", "modalidad": "L", "valor": 1e6, "proveedor": f"P{i % 17}"}
        for i in range(60)
    ]
    d = pl.enriquecer(construir_df(filas))
    assert "**17 proveedores distintos**" in lc.lectura_pareto(d).que_dice


def test_la_coincidencia_cita_los_tres_conteos(d):
    dice = lc.lectura_coincidencia(d).que_dice
    ambos = int((d["es_anomalo"] & d["reglas_riesgo"]).sum())
    solo_m = int((d["es_anomalo"] & ~d["reglas_riesgo"]).sum())
    solo_r = int((~d["es_anomalo"] & d["reglas_riesgo"]).sum())
    assert f"**{lc._miles(ambos)} contratos**" in dice
    assert f"**{lc._miles(solo_m)}**" in dice
    assert f"**{lc._miles(solo_r)}**" in dice


def test_los_umbrales_citados_son_los_calculados(d):
    dice = lc.lectura_umbrales(d).que_dice
    assert pl.fmt_cop(d.attrs["umbral_valor"]) in dice
    assert f"{d.attrs['umbral_directa'] * 100:.0f}%" in dice


def test_si_falta_un_umbral_los_demas_se_siguen_explicando(d):
    """
    El universo de prueba no tiene adiciones, así que ese umbral no existe. Los
    otros dos deben explicarse igualmente en vez de perderse todos juntos.
    """
    assert pd.isna(d.attrs["umbral_dias"])
    dice = lc.lectura_umbrales(d).que_dice
    assert "caro" in dice and "intensiva en contratación directa" in dice
    assert "adiciones largas" not in dice


def test_sin_ningun_umbral_se_dice_claramente():
    vacio = pd.DataFrame()
    vacio.attrs.update(umbral_valor=float("nan"), umbral_dias=float("nan"),
                       umbral_directa=float("nan"))
    assert "no hay datos suficientes" in lc.lectura_umbrales(vacio).que_dice.lower()


def test_el_resumen_cubre_los_puntos_clave(d):
    puntos = lc.resumen_vista(d)
    assert len(puntos) >= 4
    texto = " ".join(puntos).lower()
    for concepto in ("contratos", "entidad", "directa", "señales de alerta"):
        assert concepto in texto


def test_el_resumen_de_una_vista_vacia_lo_dice_y_no_inventa():
    d = pl.enriquecer(universo()).iloc[0:0]
    assert lc.resumen_vista(d) == [lc.VACIO]


# --------------------------------------------------------------------------- #
# Glosario
# --------------------------------------------------------------------------- #
def test_el_glosario_define_los_terminos_que_usa_la_interfaz():
    for termino in ("Contratación directa", "Mediana", "Contrato atípico", "Pyme"):
        assert termino in lc.GLOSARIO


def test_las_definiciones_del_glosario_son_frases_completas():
    for termino, definicion in lc.glosario_ordenado():
        assert len(definicion) > 40, termino
        assert definicion.strip().endswith("."), termino


def test_el_glosario_desactiva_la_lectura_acusatoria():
    """El encuadre responsable es parte del producto, no un adorno."""
    assert "no" in lc.GLOSARIO["Contrato atípico"].lower()
    assert "irregular" in lc.GLOSARIO["Contrato atípico"].lower()


def test_la_explicacion_de_contratacion_directa_no_la_presenta_como_ilegal():
    texto = lc.lectura_modalidades(pl.enriquecer(universo())).por_que_importa.lower()
    assert "la ley permite" in texto
    assert "no es ilegal" in texto


# --------------------------------------------------------------------------- #
# Render en Markdown
# --------------------------------------------------------------------------- #
def test_el_signo_de_peso_se_escapa_para_no_volverse_una_formula():
    """
    Streamlit lee `$…$` como LaTeX. Sin escapar, «vale $137 M ... vale $20 M»
    se pintaba como una fórmula matemática ilegible en lugar de como dinero.
    """
    assert lc.para_markdown("vale $137 M y $20 M") == r"vale \$137 M y \$20 M"
    assert lc.para_markdown("sin importes") == "sin importes"


@pytest.mark.parametrize("fn", TODAS, ids=lambda f: f.__name__)
def test_ninguna_lectura_deja_pares_de_pesos_sin_escapar(fn, d):
    for texto in (fn(d).que_es, fn(d).que_dice, fn(d).por_que_importa):
        pintado = lc.para_markdown(texto)
        assert "$" not in pintado.replace(r"\$", "")
