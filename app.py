# -*- coding: utf-8 -*-
"""
Observatorio ciudadano de contratación pública · Cartagena

Interfaz Streamlit sobre el núcleo analítico de `pipeline.py`: KPIs, panorama
del gasto, priorización de contratos por reglas y detección de anomalías con
Isolation Forest, presentados para público no técnico.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lecturas as lc
import pipeline as pl
from pipeline import ErrorDeDatos, abreviar, fmt_cop

# --------------------------------------------------------------------------- #
# Configuración de la página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Observatorio de contratación · Cartagena",
    page_icon="🔎",
    layout="wide",
    # "auto" la deja abierta en escritorio y plegada en pantallas pequeñas. Con
    # "expanded", quien entra desde el teléfono aterriza en el panel de filtros
    # tapando la pantalla completa, en vez de en el observatorio.
    initial_sidebar_state="auto",
)

# Paleta (consistente con las gráficas del análisis)
AZUL = "#2f6fd1"
VERDE = "#2ca25f"
NARANJA = "#e8552d"
MORADO = "#4b3f9e"
AMBAR = "#f0a500"
GRIS = "#8a8f98"
TINTA = "#12233b"

# --------------------------------------------------------------------------- #
# Estilos: identidad "control ciudadano" (institucional pero cercano)
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
      h1, h2, h3, .obs-title { font-family: 'Fraunces', Georgia, serif; letter-spacing:-.01em; }

      .obs-header { padding: .2rem 0 .1rem 0; }
      .obs-title  { font-size: 2.05rem; font-weight:600; color:#12233b; line-height:1.05; margin:0; }
      .obs-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem;
                    font-weight:600; color:#2f6fd1; margin-bottom:.35rem; }
      .obs-lead   { color:#41505f; font-size:1.02rem; max-width:70ch; margin:.35rem 0 0 0; }

      /* Tarjetas KPI */
      .kpi { background:#ffffff; border:1px solid #e7ebf0; border-radius:14px;
             padding:1.0rem 1.1rem; height:100%; }
      .kpi .lbl { font-size:.74rem; text-transform:uppercase; letter-spacing:.12em;
                  color:#8a8f98; font-weight:600; }
      .kpi .val { font-family:'Fraunces',serif; font-size:1.7rem; font-weight:600;
                  color:#12233b; margin-top:.15rem; line-height:1; }
      .kpi .sub { font-size:.8rem; color:#6b7684; margin-top:.25rem; }

      /* Aviso responsable */
      .aviso { background:#fff7ed; border:1px solid #f4cfa8; border-left:4px solid #e8552d;
               border-radius:10px; padding:.8rem 1rem; color:#7a3b16; font-size:.9rem; }

      /* Rejilla de KPIs. Son seis, asi que los cortes se eligen para que las
         filas queden siempre completas: 6, luego 3+3, luego 2+2+2. Dejarlo a
         `auto-fit` producia un feo 5+1 en pantallas intermedias. */
      .kpi-grid { display:grid; gap:.7rem;
                  grid-template-columns:repeat(6, minmax(0, 1fr)); }
      @media (max-width: 1200px) {
        .kpi-grid { grid-template-columns:repeat(3, minmax(0, 1fr)); }
      }

      /* En Cartagena la mayoria entra por el telefono. */
      @media (max-width: 640px) {
        .obs-title { font-size:1.5rem; }
        .obs-lead  { font-size:.95rem; }
        .kpi       { padding:.7rem .8rem; border-radius:12px; }
        .kpi .val  { font-size:1.3rem; }
        .kpi .lbl  { font-size:.68rem; letter-spacing:.08em; }
        .kpi-grid  { grid-template-columns:repeat(2, minmax(0, 1fr)); gap:.5rem; }
        .aviso     { font-size:.86rem; padding:.7rem .8rem; }
      }

      .stTabs [data-baseweb="tab-list"] { gap: .3rem; }
      .stTabs [data-baseweb="tab"] { font-weight:600; }
      hr { margin:.6rem 0 1rem 0; border:none; border-top:1px solid #e7ebf0; }
      [data-testid="stMetricValue"] { font-family:'Fraunces',serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Carga (con caché) y manejo de fallos
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cargando datos de contratación…")
def cargar():
    return pl.cargar_datos()


@st.cache_data(show_spinner="Calculando señales de riesgo y anomalías…")
def enriquecer(_df: pd.DataFrame, huella: str) -> pd.DataFrame:
    # `_df` no se hashea (es grande y Streamlit no puede resumirlo barato); la
    # huella del dataset sí, y es lo que identifica de verdad a esta versión de
    # los datos. Si el parquet cambia, la huella cambia y la caché se invalida.
    return pl.enriquecer(_df)


def plotly_base(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color=TINTA),
        title=dict(font=dict(family="Fraunces, serif", size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        hoverlabel=dict(font_size=12),
        # Plotly formatea por su cuenta los números de las etiquetas emergentes:
        # primero el separador decimal, después el de miles. A la colombiana.
        separators=",.",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5", zeroline=False)
    return fig


# Encabezados cortos y anchos explícitos. Con los nombres originales del
# dataset, la tabla se iba mucho más allá del ancho de la pantalla y las dos
# columnas más útiles —el nivel de riesgo y el enlace al contrato— quedaban
# fuera de la vista, donde nadie las encuentra.
COLUMNAS_TABLA = {
    "url_secop": st.column_config.LinkColumn(
        "Contrato",
        display_text="Ver en SECOP ↗",
        help="Abre la ficha oficial del proceso en el SECOP II.",
        width="small",
    ),
    "Referencia del Contrato": st.column_config.TextColumn("Referencia", width="small"),
    "Nombre Entidad": st.column_config.TextColumn("Entidad", width="medium"),
    "Valor del Contrato": st.column_config.TextColumn("Valor", width="small"),
    "Dias adicionados": st.column_config.TextColumn("Días añadidos", width="small"),
    "Modalidad de Contratacion": st.column_config.TextColumn("Modalidad", width="medium"),
}


def guia(lectura: lc.Lectura) -> None:
    """
    Icono que abre la explicación de un apartado en lenguaje llano.

    Va debajo de cada gráfica en vez de al lado del título: quien ya entiende lo
    que ve no tropieza con ella, y quien no, la encuentra justo donde le surge
    la duda.
    """
    with st.popover("¿Qué significa esto?", icon="💡"):
        st.markdown("**Qué estás viendo**")
        st.markdown(lc.para_markdown(lectura.que_es))
        st.markdown("**Qué dicen estos datos**")
        st.markdown(lc.para_markdown(lectura.que_dice))
        if lectura.por_que_importa:
            st.markdown("**Por qué importa**")
            st.markdown(lc.para_markdown(lectura.por_que_importa))


try:
    df_full, fuente = cargar()
except ErrorDeDatos as e:
    st.error(f"**No se pudieron cargar los datos de contratación.**\n\n{e}")
    st.caption(
        "La app busca primero un archivo `secop2_limpio.parquet` junto a ella y, "
        "si no está, lo descarga del repositorio de datos. Puedes apuntar a otra "
        "fuente con las variables de entorno `SECOP_DATA_URL` o `SECOP_DATA_REF`."
    )
    st.stop()

data = enriquecer(df_full, fuente.huella)

# --------------------------------------------------------------------------- #
# Barra lateral: filtros
# --------------------------------------------------------------------------- #
# Los filtros viajan en la dirección de la página. Quien encuentre algo puede
# copiar el enlace y quien lo reciba verá exactamente la misma vista: en una
# herramienta de control ciudadano, poder decir "mira esto" es media utilidad.
qp = st.query_params


def _guardado(clave, validos):
    """Valores de la URL que siguen siendo opciones válidas del filtro."""
    permitidos = set(validos)
    return [v for v in qp.get_all(clave) if v in permitidos]


def _sembrar(clave, valor):
    """
    Deja el valor de la URL en el estado del widget, **solo la primera vez**.

    Pasarlo en cada repintado como `value=`/`default=` crea un bucle: al borrar
    el buscador, la dirección todavía llevaba el término, Streamlit reconstruía
    la caja con él y el filtro volvía solo. Sembrando una vez, el widget pasa a
    ser el dueño de su estado y la URL únicamente lo refleja.
    """
    if clave not in st.session_state:
        st.session_state[clave] = valor


modos = sorted(data["Modalidad de Contratacion"].dropna().astype(str).unique())
# Ordenadas por volumen para que las grandes aparezcan primero; el buscador del
# desplegable permite llegar a cualquiera, así que no se recorta la lista.
entidades = data["Nombre Entidad"].value_counts().index.tolist()
a_min, a_max = pl.rango_anios(data)

if a_max > a_min:
    # Un año fuera de rango en la URL (enlace viejo, dataset nuevo) no debe
    # romper la página: se recorta a lo que existe.
    try:
        _desde = min(max(int(qp.get("desde", a_min)), a_min), a_max)
        _hasta = min(max(int(qp.get("hasta", a_max)), a_min), a_max)
    except (TypeError, ValueError):
        _desde, _hasta = a_min, a_max
    if _desde > _hasta:
        _desde, _hasta = a_min, a_max

_sembrar("busqueda", qp.get("q", ""))
_sembrar("modalidad", _guardado("modalidad", modos))
_sembrar("entidad", _guardado("entidad", entidades))
_sembrar("rango", _guardado("rango", pl.RANGOS_VALOR))
_sembrar("riesgo", qp.get("riesgo") == "1")
if a_max > a_min:
    _sembrar("anios", (_desde, _hasta))

# La búsqueda va primero porque suele ser lo primero que alguien quiere hacer:
# mirar una empresa concreta de la que oyó hablar, o su propio barrio.
st.sidebar.header("Buscar")
consulta = st.sidebar.text_input(
    "Empresa, entidad o número de contrato",
    key="busqueda",
    placeholder="Ej.: constructora, hospital, alcaldía…",
    help="No hace falta poner tildes ni mayúsculas, ni escribir el nombre completo.",
)

st.sidebar.header("Filtros")
st.sidebar.caption(
    "Las anomalías y las señales de riesgo se calculan sobre **todo** el "
    "universo; los filtros solo eligen qué contratos ver."
)

sel_modo = st.sidebar.multiselect(
    "Modalidad de contratación",
    modos,
    key="modalidad",
    placeholder="Todas las modalidades",
)
sel_ent = st.sidebar.multiselect(
    "Entidad",
    entidades,
    key="entidad",
    placeholder="Todas las entidades",
)

if a_max > a_min:
    sel_anios = st.sidebar.slider("Año de firma", a_min, a_max, key="anios")
else:
    # Un deslizador necesita un rango: si todos los contratos son del mismo año
    # (o no hay fechas de firma) se informa y no se ofrece el filtro.
    sel_anios = None
    if a_max > 0:
        st.sidebar.caption(f"Año de firma: todos los contratos son de {a_min}.")

sel_rango = st.sidebar.multiselect(
    "Rango de valor",
    pl.RANGOS_VALOR,
    key="rango",
    placeholder="Todos los tamaños",
)
solo_riesgo = st.sidebar.checkbox(
    "Solo contratos con señales de riesgo (2+)", key="riesgo"
)

# Reflejar la selección en la dirección de la página. Se escribe solo si cambió,
# para no reescribir la URL en cada repintado.
_url = {}
if consulta.strip():
    _url["q"] = consulta.strip()
if sel_modo:
    _url["modalidad"] = sel_modo
if sel_ent:
    _url["entidad"] = sel_ent
if sel_anios is not None and tuple(sel_anios) != (a_min, a_max):
    _url["desde"], _url["hasta"] = str(sel_anios[0]), str(sel_anios[1])
if sel_rango:
    _url["rango"] = sel_rango
if solo_riesgo:
    _url["riesgo"] = "1"

_actual = {k: qp.get_all(k) for k in qp.keys()}
_deseado = {k: (v if isinstance(v, list) else [v]) for k, v in _url.items()}
if _actual != _deseado:
    # `from_dict({})` vacía el estado interno pero no avisa al navegador, así que
    # la dirección se quedaba con el último filtro incluso después de quitarlo:
    # quien copiara el enlace compartiría una búsqueda que ya había borrado.
    if _url:
        st.query_params.from_dict(_url)
    else:
        st.query_params.clear()

# Aplicar filtros
f = pl.filtrar_por_texto(data, consulta)
if sel_modo:
    f = f[f["Modalidad de Contratacion"].astype(str).isin(sel_modo)]
if sel_ent:
    f = f[f["Nombre Entidad"].isin(sel_ent)]
if sel_anios is not None:
    f = f[f["anio_firma"].isna() | f["anio_firma"].between(*sel_anios)]
if sel_rango:
    f = f[f["Rango Valor"].astype(str).isin(sel_rango)]
if solo_riesgo:
    f = f[f["reglas_riesgo"]]

n_contratos = f["ID Contrato"].nunique()

st.sidebar.markdown("---")
st.sidebar.metric("Contratos en la vista", pl.fmt_entero(n_contratos))
if n_contratos != len(f):
    st.sidebar.caption(
        f"⚠️ {pl.fmt_entero(len(f))} filas para {pl.fmt_entero(n_contratos)} contratos distintos: "
        "el dataset trae registros repetidos."
    )
if consulta.strip() and n_contratos == 0:
    st.sidebar.warning(
        f"Nada coincide con «{consulta.strip()}». Prueba con menos palabras o con "
        "una parte del nombre."
    )

st.sidebar.caption("Fuente: SECOP II (Colombia Compra Eficiente), datos abiertos.")
st.sidebar.caption(f"Datos: `{fuente.origen}` · versión `{fuente.huella}`")

# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="obs-header">
      <div class="obs-kicker">Control social del gasto público · Cartagena</div>
      <div class="obs-title">Observatorio ciudadano de contratación pública</div>
      <p class="obs-lead">Convierte los datos abiertos de contratación del Distrito de Cartagena
      en información comprensible: dónde se concentra el gasto, cómo se ejecuta y qué contratos
      ameritan una mirada más de cerca.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Busca una empresa o entidad y acota por año o tamaño del contrato desde la barra "
    "lateral (en el teléfono se abre con la flecha de arriba a la izquierda). El icono "
    "💡 de cada apartado explica qué estás viendo. La dirección de la página guarda lo "
    "que filtres: cópiala para compartir exactamente esta vista."
)
st.markdown("<hr/>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# KPIs (sobre la vista filtrada)
# --------------------------------------------------------------------------- #
valor_total = f["Valor del Contrato"].sum()
valor_prom = f["Valor del Contrato"].mean() if len(f) else np.nan
duracion = f["duracion_dias"].mean() if len(f) else np.nan
pyme_pct = f["Es Pyme"].mean() * 100 if len(f) else np.nan
n_atipicos = int(f["es_anomalo"].sum())

kpis = [
    ("Contratos", pl.fmt_entero(n_contratos), "en la vista actual"),
    ("Valor total", fmt_cop(valor_total), "suma contratada"),
    ("Valor promedio", fmt_cop(valor_prom), "por contrato"),
    ("Duración media", f"{duracion:.0f} d" if pd.notna(duracion) else "—", "inicio → fin"),
    ("Participación Pyme", f"{pyme_pct:.0f}%" if pd.notna(pyme_pct) else "—", "de los contratos"),
    ("Contratos atípicos", pl.fmt_entero(n_atipicos), "detectados por el modelo"),
]
_tarjetas = "".join(
    '<div class="kpi">'
    f'<div class="lbl">{lbl}</div>'
    f'<div class="val">{val}</div>'
    f'<div class="sub">{sub}</div>'
    "</div>"
    for lbl, val, sub in kpis
)
st.markdown(f'<div class="kpi-grid">{_tarjetas}</div>', unsafe_allow_html=True)

guia(lc.lectura_kpis(f))

# --------------------------------------------------------------------------- #
# Resumen en palabras: lo que debería bastar si alguien solo lee una cosa
# --------------------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 🧭 Esto es lo que estás viendo, en pocas palabras")
    for punto in lc.resumen_vista(f):
        st.markdown(f"- {lc.para_markdown(punto)}")

st.markdown("<br/>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Pestañas
# --------------------------------------------------------------------------- #
tab_pan, tab_prio, tab_metodo = st.tabs(
    ["📊  Panorama del gasto", "🔎  Contratos priorizados", "📖  Metodología ciudadana"]
)

# ============================ PANORAMA ===================================== #
with tab_pan:
    if len(f) == 0:
        st.info("No hay contratos con los filtros actuales. Ajusta los filtros de la izquierda.")
    else:
        c1, c2 = st.columns(2)

        # 1) Entidades que más contratan (por monto)
        with c1:
            top = (f.groupby("Nombre Entidad", observed=True)["Valor del Contrato"]
                     .sum().sort_values(ascending=False).head(6) / 1e9)
            fig = go.Figure(go.Bar(
                x=top.values, y=abreviar(top.index, 38), orientation="h",
                marker_color=AZUL,
                text=[f"${pl.fmt_entero(v)}" for v in top.values], textposition="outside",
                customdata=list(top.index),
                hovertemplate="%{customdata}<br>$%{x:,.0f} mil millones<extra></extra>",
                cliponaxis=False,
            ))
            fig.update_layout(title="Entidades que más contratan (por monto)",
                              xaxis_title="Mil millones COP")
            fig.update_yaxes(autorange="reversed")
            # Sitio para la etiqueta de la barra mas larga: sin este margen,
            # justo la cifra mas importante queda cortada contra el borde.
            fig.update_xaxes(range=[0, float(top.max()) * 1.32])
            st.plotly_chart(plotly_base(fig), width="stretch")
            guia(lc.lectura_entidades(f))

        # 2) Distribución por modalidad (dona)
        with c2:
            modal = f["Modalidad de Contratacion"].value_counts()
            # value_counts sobre una columna categórica devuelve también las
            # categorías que no tienen ningún contrato en la vista: se descartan.
            modal = modal[modal > 0]
            if len(modal) > 5:
                otras = modal.iloc[5:].sum()
                modal = modal.iloc[:5]
                modal["Otras"] = otras
            # Solo se rotulan las porciones con sitio suficiente. Rotularlas todas
            # amontonaba etiquetas de tajadas del 1% con lineas guia cruzadas, y
            # una salia con tres decimales. Las pequenas se leen en la leyenda o
            # pasando el cursor por encima.
            _porcion = modal.values / modal.values.sum() * 100
            _rotulos = [f"{v:.1f}%".replace(".", ",") if v >= 4 else "" for v in _porcion]
            fig = go.Figure(go.Pie(
                labels=modal.index.astype(str), values=modal.values, hole=0.58,
                marker=dict(colors=[AZUL, VERDE, AMBAR, MORADO, NARANJA, GRIS]),
                text=_rotulos, textinfo="text", textposition="inside",
                insidetextfont=dict(color="white", size=13),
                sort=False,
                hovertemplate="%{label}<br>%{value:,} contratos (%{percent})<extra></extra>",
            ))
            fig.update_layout(title="Distribución por modalidad")
            st.plotly_chart(plotly_base(fig), width="stretch")
            guia(lc.lectura_modalidades(f))

        c3, c4 = st.columns(2)

        # 3) Evolución del valor por año
        with c3:
            por_anio = (f.dropna(subset=["Fecha de Firma"])
                          .groupby("anio_firma")["Valor del Contrato"].sum() / 1e9)
            fig = go.Figure(go.Bar(
                x=por_anio.index.astype(int).astype(str), y=por_anio.values,
                marker_color=AZUL, text=[f"${pl.fmt_entero(v)}" for v in por_anio.values],
                textposition="outside",
                hovertemplate="%{x}<br>$%{y:,.0f} mil millones<extra></extra>",
            ))
            fig.update_layout(title="Evolución del valor contratado por año",
                              yaxis_title="Mil millones COP")
            st.plotly_chart(plotly_base(fig), width="stretch")
            guia(lc.lectura_evolucion(f))

        # 4) Valor mediano por modalidad (robusto frente a los extremos)
        with c4:
            resumen = (f.groupby("Modalidad de Contratacion", observed=True)["Valor del Contrato"]
                         .agg(["median", "count"]))
            resumen = (resumen[resumen["count"] >= 30]
                       .sort_values("median", ascending=False).head(6))
            med = resumen["median"] / 1e6
            if med.empty:
                st.info("Ninguna modalidad alcanza los 30 contratos en esta vista.")
            else:
                fig = go.Figure(go.Bar(
                    x=abreviar(med.index, 22), y=med.values, marker_color=VERDE,
                    text=[f"${pl.fmt_entero(v)}" for v in med.values], textposition="outside",
                    customdata=list(med.index.astype(str)),
                    hovertemplate="%{customdata}<br>$%{y:,.0f} millones (mediana)<extra></extra>",
                ))
                fig.update_layout(title="Valor mediano por modalidad (top 6)",
                                  yaxis_title="Millones COP")
                fig.update_xaxes(tickangle=-20)
                st.plotly_chart(plotly_base(fig), width="stretch")
                guia(lc.lectura_valor_mediano(f))

        c5, c6 = st.columns(2)

        # 5) Acceso Pyme por rango de valor
        with c5:
            tabla = (f.groupby("Rango Valor", observed=True)["Es Pyme"].mean() * 100)
            tabla = tabla.reindex([r for r in pl.RANGOS_VALOR if r in tabla.index]).dropna()
            if tabla.empty:
                st.info("No hay datos de participación Pyme en esta vista.")
            else:
                fig = go.Figure()
                fig.add_bar(x=tabla.index.astype(str), y=tabla.values, name="Pyme",
                            marker_color=VERDE,
                            text=[f"{v:.0f}%" for v in tabla.values], textposition="inside")
                fig.add_bar(x=tabla.index.astype(str), y=(100 - tabla.values), name="No Pyme",
                            marker_color=MORADO)
                fig.update_layout(title="Acceso Pyme por rango de valor", barmode="stack",
                                  yaxis_title="% de contratos", yaxis_range=[0, 100])
                st.plotly_chart(plotly_base(fig), width="stretch")
                guia(lc.lectura_pyme(f))

        # 6) Concentración de proveedores (Pareto)
        with c6:
            prov = (f.groupby("Proveedor Adjudicado", observed=True)["Valor del Contrato"]
                      .sum().sort_values(ascending=False))
            top = prov.head(6)
            pareto = top
            # Solo tiene sentido añadir "Otros" si de verdad quedan más proveedores.
            if len(prov) > 6:
                pareto = pd.concat([top, pd.Series({"Otros": prov.iloc[6:].sum()})])
            pct = pareto / pareto.sum() * 100
            pct_acum = pct.cumsum()
            # `abreviar` en vez de un recorte simple: 761 de los 28.206 proveedores
            # coinciden en sus primeros 18 caracteres, y Plotly fundiría en una
            # sola barra a dos que se truncaran igual.
            etiquetas = abreviar(pct.index, 18)
            fig = go.Figure()
            fig.add_bar(x=etiquetas, y=pct.values, marker_color=AZUL, name="% del monto",
                        customdata=list(pct.index.astype(str)),
                        hovertemplate="%{customdata}<br>%{y:.1f}% del monto<extra></extra>")
            fig.add_trace(go.Scatter(x=etiquetas, y=pct_acum.values, name="% acumulado",
                                     mode="lines+markers", line=dict(color=NARANJA, width=2),
                                     customdata=list(pct.index.astype(str)),
                                     hovertemplate="%{customdata}<br>%{y:.1f}% acumulado<extra></extra>",
                                     yaxis="y2"))
            fig.update_layout(
                title="Concentración de proveedores (Pareto)",
                yaxis=dict(title="% del monto"),
                yaxis2=dict(title="% acumulado", overlaying="y", side="right",
                            range=[0, 105], showgrid=False),
            )
            fig.update_xaxes(tickangle=-25)
            fig = plotly_base(fig)
            # Este ajuste va DESPUÉS de plotly_base, que impone la leyenda al pie
            # y un margen inferior mínimo. Aquí abajo van los nombres de los
            # proveedores inclinados, así que la leyenda se sube al encabezado y
            # se reserva sitio para ellos; de lo contrario se solapan.
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(l=10, r=10, t=60, b=100),
            )
            st.plotly_chart(fig, width="stretch")
            n_top = min(6, len(prov))
            st.caption(
                f"De **{pl.fmt_entero(len(prov))} proveedores** contratados, los {n_top} "
                f"mayores concentran el **{pct.head(6).sum():.0f}%** del monto (vista actual)."
            )
            guia(lc.lectura_pareto(f))

# ======================= CONTRATOS PRIORIZADOS ============================= #
with tab_prio:
    st.markdown(
        '<div class="aviso"><b>Cómo leer esta sección.</b> Un contrato “atípico” o '
        '“priorizado” es estadísticamente inusual, <b>no</b> necesariamente irregular. '
        "Estas listas señalan <b>dónde mirar</b>, no quién falló.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    uv = data.attrs["umbral_valor"]
    ud = data.attrs["umbral_dias"]
    udir = data.attrs["umbral_directa"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Umbral de valor (p95)", fmt_cop(uv))
    m2.metric(
        "Umbral de adiciones (p90 de >0)",
        f"{pl.fmt_entero(ud)} días" if pd.notna(ud) else "—",
    )
    m3.metric(
        "Umbral de contratación directa",
        f"{udir * 100:.0f}%" if pd.notna(udir) else "—",
        help=(
            "Una entidad se considera intensiva en contratación directa cuando la usa "
            "en más de este porcentaje de sus contratos: es el percentil 75 entre las "
            f"{data.attrs['entidades_evaluables']} entidades con al menos "
            f"{pl.MIN_CONTRATOS_ENTIDAD} contratos."
        ),
    )
    guia(lc.lectura_umbrales(data))

    st.subheader("Priorización por reglas")
    st.caption(
        "Combina tres señales objetivas: valor en el 5% más alto, pertenecer a una "
        "entidad que recurre a la contratación directa más que sus pares, y adiciones "
        "de plazo extensas. Más señales encendidas = mayor prioridad de revisión."
    )
    guia(lc.lectura_reglas(f))
    reglas = (f[f["senales"] >= 2]
              .sort_values(["senales", "Valor del Contrato"], ascending=False))
    if reglas.empty:
        st.info("Ningún contrato acumula 2 o más señales en esta vista.")
    else:
        vista_r = reglas[["Riesgo", "Referencia del Contrato", "url_secop",
                          "Nombre Entidad", "Valor del Contrato", "Dias adicionados",
                          "Modalidad de Contratacion"]].copy()
        vista_r["Valor del Contrato"] = vista_r["Valor del Contrato"].apply(fmt_cop)
        vista_r["Dias adicionados"] = (
            "+" + vista_r["Dias adicionados"].fillna(0).astype(int).astype(str) + " d"
        )
        st.dataframe(vista_r.head(50), width="stretch", hide_index=True,
                     column_config=COLUMNAS_TABLA)
        st.caption(
            f"{pl.fmt_entero(len(reglas))} contratos con 2+ señales · "
            f"{pl.fmt_entero(int((f['senales'] == 3).sum()))} críticos (3 señales) en la vista. "
            "Se muestran los 50 primeros."
        )

    st.markdown("---")
    st.subheader("Detección de anomalías (Isolation Forest)")
    st.caption(
        "Un modelo de aprendizaje no supervisado que marca contratos atípicos "
        "combinando valor, duración, adiciones y modalidad, sin reglas predefinidas."
    )
    guia(lc.lectura_anomalias(f))
    anom = f[f["es_anomalo"]].sort_values("anomaly_score")
    if anom.empty:
        st.info("El modelo no marcó ningún contrato atípico en esta vista.")
    else:
        vista_a = anom[["Referencia del Contrato", "url_secop", "anomaly_score",
                        "Nombre Entidad", "Valor del Contrato", "Dias adicionados",
                        "Modalidad de Contratacion"]].copy()
        vista_a["Valor del Contrato"] = vista_a["Valor del Contrato"].apply(fmt_cop)
        vista_a["Dias adicionados"] = vista_a["Dias adicionados"].fillna(0).astype(int)
        col_at = "Atipicidad"
        vista_a = vista_a.rename(columns={"anomaly_score": col_at})
        vista_a[col_at] = vista_a[col_at].round(3)
        st.dataframe(
            vista_a.head(50), width="stretch", hide_index=True,
            column_config=COLUMNAS_TABLA
            | {col_at: st.column_config.NumberColumn(
                col_at, width="small",
                help="Cuanto más bajo, más se aparta del patrón general.")},
        )
        st.caption(f"{pl.fmt_entero(len(anom))} contratos atípicos en la vista. Se muestran los 50 primeros.")

    # Contraste modelo vs reglas
    st.markdown("---")
    st.subheader("¿Coinciden el modelo y las reglas?")
    guia(lc.lectura_coincidencia(f))
    ambos = int((f["es_anomalo"] & f["reglas_riesgo"]).sum())
    solo_m = int((f["es_anomalo"] & ~f["reglas_riesgo"]).sum())
    solo_r = int((~f["es_anomalo"] & f["reglas_riesgo"]).sum())
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Detectados por ambos", pl.fmt_entero(ambos), help="Se validan mutuamente")
    cc2.metric("Solo por el modelo", pl.fmt_entero(solo_m), help="Hallazgos que las reglas no vieron")
    cc3.metric("Solo por las reglas", pl.fmt_entero(solo_r), help="El modelo no los marcó")

    n_reglas = ambos + solo_r
    coincidencia = (ambos / n_reglas * 100) if n_reglas else np.nan
    st.caption(
        "Los dos métodos **no** buscan lo mismo, y esa es la idea. Las reglas son "
        "deliberadamente estrictas: exigen que se acumulen al menos dos señales, así que "
        "producen una lista corta y muy defendible. El modelo es más amplio: marca un "
        f"porcentaje fijo del universo ({pl.CONTAMINATION:.0%}) buscando combinaciones "
        "inusuales, aunque ninguna señal individual se dispare."
        + (
            f" En esta vista, **{coincidencia:.0f}%** de los contratos señalados por las "
            "reglas también aparecen para el modelo: cuando ambos coinciden, la señal es "
            "más fuerte; los que solo ve el modelo son casos que ninguna regla fija habría "
            "encontrado."
            if pd.notna(coincidencia) else ""
        )
    )

    # Descarga
    st.markdown("---")
    export = f.loc[
        f["reglas_riesgo"] | f["es_anomalo"],
        ["Referencia del Contrato", "Nombre Entidad", "Valor del Contrato",
         "Dias adicionados", "Modalidad de Contratacion", "Riesgo",
         "flag_valor", "flag_directa", "flag_adiciones",
         "tasa_directa_entidad", "es_anomalo", "anomaly_score", "url_secop"],
    ]
    st.download_button(
        "⬇️  Descargar contratos priorizados (CSV)",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name="contratos_priorizados_cartagena.csv",
        mime="text/csv",
        disabled=export.empty,
    )
    st.caption(f"{pl.fmt_entero(len(export))} contratos en la descarga (todos los de la vista, no solo los 50 mostrados).")

# ======================== METODOLOGÍA CIUDADANA ============================ #
with tab_metodo:
    st.subheader("¿Qué estás viendo y de dónde sale?")
    st.markdown(
        f"""
Este observatorio parte de los **datos abiertos** del **SECOP II**, la plataforma
donde las entidades públicas registran su contratación (administrada por Colombia
Compra Eficiente). Tomamos los contratos del Distrito de Cartagena, los **limpiamos**
y los presentamos aquí de forma navegable.

**Cómo priorizamos contratos.** Usamos dos enfoques que se complementan:

- **Reglas transparentes.** Marcamos un contrato cuando cumple señales objetivas:
  está entre el **{1 - pl.PCT_VALOR_ALTO:.0%} de mayor valor**; pertenece a una entidad que
  **recurre a la contratación directa más que sus pares**; o tiene **adiciones de plazo**
  inusualmente largas. Los cortes no son arbitrarios: salen de los propios datos
  (percentiles).

- **Un modelo que aprende solo (Isolation Forest).** Sin decirle qué buscar, el modelo
  detecta contratos que se salen del patrón general combinando varias características a la
  vez. Sirve para encontrar casos raros que una sola regla no vería.

Cuando **ambos** métodos coinciden en un contrato, la señal es más fuerte. Cuando difieren,
cada uno aporta una mirada distinta.
        """
    )

    st.markdown("#### Por qué la contratación directa se mide contra los pares")
    st.markdown(
        f"""
En Cartagena la contratación directa **es la norma, no la excepción**: alrededor de
**{data['es_directa'].mean():.0%}** de los contratos del universo se adjudican así. Marcar
un contrato solo por ser directo dejaría a cinco de cada seis "señalados", y una alerta que
suena siempre no informa nada.

Por eso la señal no pregunta *¿este contrato fue directo?* sino *¿la entidad que lo firmó
recurre a la contratación directa más que entidades comparables?*. Se considera intensiva a
la que supera el **percentil {pl.PCT_DIRECTA_PARES:.0%}** de sus pares, y solo se evalúan las
**{data.attrs['entidades_evaluables']} de {data.attrs['entidades_totales']} entidades** con al
menos **{pl.MIN_CONTRATOS_ENTIDAD} contratos**, porque por debajo de ese volumen el porcentaje
es demasiado inestable para juzgar a nadie con él. Las entidades pequeñas nunca se marcan por
esta vía: preferimos no señalar a señalar sin base.
        """
    )

    st.markdown("#### Qué mira exactamente el modelo")
    st.markdown(
        f"""
El Isolation Forest (`contamination={pl.CONTAMINATION}`, `n_estimators={pl.N_ESTIMATORS}`,
`random_state={pl.RANDOM_STATE}`) trabaja con cuatro variables: valor, duración, días
adicionados y si la modalidad fue directa.

El valor de los contratos abarca varios órdenes de magnitud, así que entra **transformado a
escala logarítmica**. Sin esa transformación el modelo se limitaba a redescubrir "los
contratos más caros" —que es justo lo que ya detecta la regla del valor— en vez de aportar
combinaciones inusuales. Con ella, el detector aporta información propia en lugar de duplicar
una regla que ya tenemos.
        """
    )

    cal = pl.resumen_calidad(data)
    st.markdown("#### Límites de los datos")
    st.markdown(
        f"""
Los datos abiertos traen imperfecciones y preferimos declararlas:

- **{pl.fmt_entero(cal['sin_duracion'])}** contratos (de {pl.fmt_entero(cal['total'])}) no tienen una duración
  calculable, porque les falta la fecha de inicio o de fin.
- **{pl.fmt_entero(cal['fechas_invertidas'])}** registran una fecha de finalización **anterior** a la de
  inicio. Es un error de captura, así que su duración se trata como dato ausente en vez de
  dejarla entrar al modelo como si fuera válida.
- El observatorio describe **lo que se registró en el SECOP II**. Un contrato mal diligenciado
  se ve mal aquí, y eso no es lo mismo que un contrato mal ejecutado.
        """
    )

    st.markdown("#### Glosario")
    st.caption(
        "Las palabras que aparecen en el tablero, explicadas sin tecnicismos. "
        "Despliega la que necesites."
    )
    for termino, definicion in lc.glosario_ordenado():
        with st.expander(termino):
            st.markdown(definicion)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        '<div class="aviso"><b>Un contrato señalado no es un contrato corrupto.</b> '
        "Ser “atípico” significa que se aparta de lo común y por eso <b>merece revisión</b>. "
        "Puede tener una explicación legítima. Esta herramienta orienta el control ciudadano; "
        "no emite juicios ni acusaciones.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Fuente de datos: SECOP II · Colombia Compra Eficiente (datos abiertos). "
        f"Versión del dataset en uso: `{fuente.huella}` ({fuente.origen}, "
        f"{fuente.n_bytes / 1e6:.1f} MB)."
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Observatorio ciudadano de contratación pública de Cartagena · base metodológica. "
    "Datos: SECOP II (datos abiertos). Las anomalías son estadísticas, no determinaciones legales."
)
