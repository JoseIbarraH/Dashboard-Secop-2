# -*- coding: utf-8 -*-
"""
Observatorio ciudadano de contratación pública · Cartagena
Base metodológica: pipeline de datos + priorización por reglas + Isolation Forest.

La app lee el dataset limpio (secop2_limpio.parquet) generado por el notebook,
calcula los KPIs, las gráficas, la priorización por reglas y el modelo de
detección de anomalías, y los expone de forma interactiva para público no técnico.
"""

import io
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuración de la página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Observatorio de contratación · Cartagena",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta (consistente con las gráficas del análisis)
AZUL = "#2f6fd1"
VERDE = "#2ca25f"
NARANJA = "#e8552d"
MORADO = "#4b3f9e"
AMBAR = "#f0a500"
GRIS = "#8a8f98"
TINTA = "#12233b"

DATA_URL = ("https://raw.githubusercontent.com/JoseIbarraH/"
            "datasets-analitica-de-datos/main/secop2_limpio.parquet")

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

      .stTabs [data-baseweb="tab-list"] { gap: .3rem; }
      .stTabs [data-baseweb="tab"] { font-weight:600; }
      hr { margin:.6rem 0 1rem 0; border:none; border-top:1px solid #e7ebf0; }
      [data-testid="stMetricValue"] { font-family:'Fraunces',serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def fmt_cop(x: float) -> str:
    """Formatea pesos: $1,24 B / $322 M / $1.234 (coma decimal COP)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x >= 1e12:
        s = f"{x/1e12:.2f} B"
    elif x >= 1e6:
        s = f"{x/1e6:.0f} M"
    else:
        s = f"{x:,.0f}"
    return "$" + s.replace(".", ",")

@st.cache_data(show_spinner="Cargando datos de contratación…")
def load_data() -> pd.DataFrame:
    """Lee el parquet local (si está junto a la app) o lo descarga de GitHub."""
    local = "secop2_limpio.parquet"
    if os.path.exists(local):
        return pd.read_parquet(local)
    with urllib.request.urlopen(DATA_URL) as r:
        raw = r.read()
    return pd.read_parquet(io.BytesIO(raw))


@st.cache_data(show_spinner="Calculando señales de riesgo y anomalías…")
def enrich(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade, sobre TODO el dataset (umbrales relativos al universo completo):
      - duracion_dias, es_directa, Rango Valor
      - señales de reglas (valor p95, adiciones p90>0, modalidad directa) y Riesgo
      - Isolation Forest: es_anomalo, anomaly_score
    Se calcula una sola vez; los filtros solo eligen qué mostrar.
    """
    d = _df.copy()

    # --- Variables derivadas -------------------------------------------------
    d["duracion_dias"] = (
        d["Fecha de Fin del Contrato"] - d["Fecha de Inicio del Contrato"]
    ).dt.days
    d["es_directa"] = (
        d["Modalidad de Contratacion"].astype(str)
        .str.contains("directa", case=False, na=False).astype(int)
    )
    bins = [0, 90e6, 500e6, 1e9, float("inf")]
    labels = ["< 90 M", "90–500 M", "500 M–1 B", "> 1 B"]
    d["Rango Valor"] = pd.cut(d["Valor del Contrato"], bins=bins, labels=labels, right=False)
    d["anio_firma"] = d["Fecha de Firma"].dt.year

    # --- Reglas (umbrales por percentiles del propio dataset) ----------------
    umbral_valor = d["Valor del Contrato"].quantile(0.95)
    adic_pos = d["Dias adicionados"][d["Dias adicionados"] > 0]
    umbral_dias = adic_pos.quantile(0.90)

    d["flag_valor"] = d["Valor del Contrato"] > umbral_valor
    d["flag_directa"] = d["es_directa"] == 1
    d["flag_adiciones"] = d["Dias adicionados"].fillna(0) > umbral_dias
    d["senales"] = d[["flag_valor", "flag_directa", "flag_adiciones"]].sum(axis=1)

    def nivel(s):
        return "Crítico" if s >= 3 else "Alto" if s == 2 else "Medio" if s == 1 else "Bajo"
    d["Riesgo"] = d["senales"].apply(nivel)
    d["reglas_riesgo"] = d["senales"] >= 2

    # --- Isolation Forest ----------------------------------------------------
    features = ["Valor del Contrato", "duracion_dias", "Dias adicionados", "es_directa"]
    X = d[features].copy()
    X["Dias adicionados"] = X["Dias adicionados"].fillna(0)
    X["duracion_dias"] = X["duracion_dias"].fillna(X["duracion_dias"].median())
    mask = X.notna().all(axis=1)

    X_scaled = StandardScaler().fit_transform(X[mask])
    iso = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
    pred = iso.fit_predict(X_scaled)
    score = iso.decision_function(X_scaled)

    d["es_anomalo"] = False
    d["anomaly_score"] = np.nan
    d.loc[mask, "es_anomalo"] = pred == -1
    d.loc[mask, "anomaly_score"] = score

    d.attrs["umbral_valor"] = umbral_valor
    d.attrs["umbral_dias"] = umbral_dias
    return d


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
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5", zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
df_full = load_data()
data = enrich(df_full)

# --------------------------------------------------------------------------- #
# Barra lateral: filtros
# --------------------------------------------------------------------------- #
st.sidebar.header("Filtros")
st.sidebar.caption(
    "Las anomalías y las señales de riesgo se calculan sobre **todo** el "
    "universo; los filtros solo eligen qué contratos ver."
)

modos = sorted(data["Modalidad de Contratacion"].dropna().astype(str).unique())
sel_modo = st.sidebar.multiselect("Modalidad de contratación", modos, default=[])

entidades = (
    data["Nombre Entidad"].value_counts().index.tolist()
)
sel_ent = st.sidebar.multiselect(
    "Entidad (deja vacío para todas)", entidades[:200], default=[]
)

anios = sorted(int(a) for a in data["anio_firma"].dropna().unique())
if anios:
    a_min, a_max = min(anios), max(anios)
    sel_anios = st.sidebar.slider(
        "Año de firma", a_min, a_max, (max(a_min, 2020), a_max)
    )
else:
    sel_anios = None

sel_rango = st.sidebar.multiselect(
    "Rango de valor", ["< 90 M", "90–500 M", "500 M–1 B", "> 1 B"], default=[]
)

solo_riesgo = st.sidebar.checkbox("Solo contratos con señales de riesgo (2+)", value=False)

# Aplicar filtros
f = data.copy()
if sel_modo:
    f = f[f["Modalidad de Contratacion"].astype(str).isin(sel_modo)]
if sel_ent:
    f = f[f["Nombre Entidad"].isin(sel_ent)]
if sel_anios is not None:
    f = f[(f["anio_firma"].isna()) | (f["anio_firma"].between(sel_anios[0], sel_anios[1]))]
if sel_rango:
    f = f[f["Rango Valor"].astype(str).isin(sel_rango)]
if solo_riesgo:
    f = f[f["reglas_riesgo"]]

st.sidebar.markdown("---")
st.sidebar.metric("Contratos en la vista", f"{len(f):,}")
st.sidebar.caption("Fuente: SECOP II (Colombia Compra Eficiente), datos abiertos.")

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
st.markdown("<hr/>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# KPIs (sobre la vista filtrada)
# --------------------------------------------------------------------------- #
n_contratos = f["ID Contrato"].nunique()
valor_total = f["Valor del Contrato"].sum()
valor_prom = f["Valor del Contrato"].mean()
duracion = (f["Fecha de Fin del Contrato"] - f["Fecha de Inicio del Contrato"]).dt.days.mean()
pyme_pct = f["Es Pyme"].mean() * 100 if len(f) else np.nan
n_atipicos = int(f["es_anomalo"].sum())

kpis = [
    ("Contratos", f"{n_contratos:,}", "en la vista actual"),
    ("Valor total", fmt_cop(valor_total), "suma contratada"),
    ("Valor promedio", fmt_cop(valor_prom), "por contrato"),
    ("Duración media", f"{duracion:.0f} d" if pd.notna(duracion) else "—", "inicio → fin"),
    ("Participación Pyme", f"{pyme_pct:.0f}%" if pd.notna(pyme_pct) else "—", "de los contratos"),
    ("Contratos atípicos", f"{n_atipicos:,}", "detectados por el modelo"),
]
cols = st.columns(6)
for c, (lbl, val, sub) in zip(cols, kpis):
    c.markdown(
        f'<div class="kpi"><div class="lbl">{lbl}</div>'
        f'<div class="val">{val}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

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
            top.index = [n[:38] + "…" if len(n) > 38 else n for n in top.index]
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=AZUL,
                text=[f"${v:,.0f}" for v in top.values], textposition="outside",
                hovertemplate="%{y}<br>$%{x:,.0f} mil millones<extra></extra>",
            ))
            fig.update_layout(title="Entidades que más contratan (por monto)",
                              xaxis_title="Mil millones COP")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(plotly_base(fig), width='stretch')

        # 2) Distribución por modalidad (dona)
        with c2:
            modal = f["Modalidad de Contratacion"].value_counts()
            if len(modal) > 5:
                otras = modal.iloc[5:].sum()
                modal = modal.iloc[:5]
                modal["Otras"] = otras
            fig = go.Figure(go.Pie(
                labels=modal.index.astype(str), values=modal.values, hole=0.58,
                marker=dict(colors=[AZUL, VERDE, AMBAR, MORADO, NARANJA, GRIS]),
                textinfo="percent", sort=False,
                hovertemplate="%{label}<br>%{value:,} contratos (%{percent})<extra></extra>",
            ))
            fig.update_layout(title="Distribución por modalidad")
            st.plotly_chart(plotly_base(fig), width='stretch')

        c3, c4 = st.columns(2)

        # 3) Evolución del valor por año
        with c3:
            por_anio = (f.dropna(subset=["Fecha de Firma"])
                          .groupby("anio_firma")["Valor del Contrato"].sum() / 1e9)
            por_anio = por_anio[(por_anio.index >= 2016) & (por_anio.index <= 2026)]
            fig = go.Figure(go.Bar(
                x=por_anio.index.astype(int).astype(str), y=por_anio.values,
                marker_color=AZUL, text=[f"${v:,.0f}" for v in por_anio.values],
                textposition="outside",
                hovertemplate="%{x}<br>$%{y:,.0f} mil millones<extra></extra>",
            ))
            fig.update_layout(title="Evolución del valor contratado por año",
                              yaxis_title="Mil millones COP")
            st.plotly_chart(plotly_base(fig), width='stretch')

        # 4) Valor mediano por modalidad (robusto)
        with c4:
            g = f.groupby("Modalidad de Contratacion", observed=True)["Valor del Contrato"]
            resumen = g.agg(["median", "count"])
            resumen = (resumen[resumen["count"] >= 30]
                       .sort_values("median", ascending=False).head(6))
            med = resumen["median"] / 1e6
            fig = go.Figure(go.Bar(
                x=[m[:22] + "…" if len(m) > 22 else m for m in med.index.astype(str)],
                y=med.values, marker_color=VERDE,
                text=[f"${v:,.0f}" for v in med.values], textposition="outside",
                hovertemplate="%{x}<br>$%{y:,.0f} millones (mediana)<extra></extra>",
            ))
            fig.update_layout(title="Valor mediano por modalidad (top 6)",
                              yaxis_title="Millones COP")
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(plotly_base(fig), width='stretch')

        c5, c6 = st.columns(2)

        # 5) Acceso Pyme por rango de valor
        with c5:
            tabla = (f.groupby("Rango Valor", observed=True)["Es Pyme"].mean() * 100)
            orden = ["< 90 M", "90–500 M", "500 M–1 B", "> 1 B"]
            tabla = tabla.reindex([o for o in orden if o in tabla.index])
            fig = go.Figure()
            fig.add_bar(x=tabla.index.astype(str), y=tabla.values, name="Pyme",
                        marker_color=VERDE,
                        text=[f"{v:.0f}%" for v in tabla.values], textposition="inside")
            fig.add_bar(x=tabla.index.astype(str), y=(100 - tabla.values), name="No Pyme",
                        marker_color=MORADO)
            fig.update_layout(title="Acceso Pyme por rango de valor", barmode="stack",
                              yaxis_title="% de contratos", yaxis_range=[0, 100])
            st.plotly_chart(plotly_base(fig), width='stretch')

        # 6) Concentración de proveedores (Pareto)
        with c6:
            prov = (f.groupby("Proveedor Adjudicado", observed=True)["Valor del Contrato"]
                      .sum().sort_values(ascending=False))
            top = prov.head(6)
            pareto = pd.concat([top, pd.Series({"Otros": prov.iloc[6:].sum()})])
            pct = pareto / pareto.sum() * 100
            pct_acum = pct.cumsum()
            etiquetas = [p[:18] + "…" if len(str(p)) > 18 else str(p) for p in pct.index]
            fig = go.Figure()
            fig.add_bar(x=etiquetas, y=pct.values, marker_color=AZUL, name="% del monto",
                        hovertemplate="%{x}<br>%{y:.1f}% del monto<extra></extra>")
            fig.add_trace(go.Scatter(x=etiquetas, y=pct_acum.values, name="% acumulado",
                                     mode="lines+markers", line=dict(color=NARANJA, width=2),
                                     yaxis="y2"))
            fig.update_layout(
                title="Concentración de proveedores (Pareto)",
                yaxis=dict(title="% del monto"),
                yaxis2=dict(title="% acumulado", overlaying="y", side="right",
                            range=[0, 105], showgrid=False),
            )
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(plotly_base(fig), width='stretch')
            top6 = pct.head(6).sum()
            st.caption(f"Los 6 mayores proveedores concentran el **{top6:.0f}%** del monto (vista actual).")

# ======================= CONTRATOS PRIORIZADOS ============================= #
with tab_prio:
    st.markdown(
        '<div class="aviso"><b>Cómo leer esta sección.</b> Un contrato “atípico” o '
        '“priorizado” es estadísticamente inusual, <b>no</b> necesariamente irregular. '
        'Estas listas señalan <b>dónde mirar</b>, no quién falló.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    uv = data.attrs["umbral_valor"]
    ud = data.attrs["umbral_dias"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Umbral de valor (p95)", fmt_cop(uv))
    m2.metric("Umbral de adiciones (p90 de >0)", f"{ud:,.0f} días")
    m3.metric("Coinciden modelo + reglas", f"{int((f['es_anomalo'] & f['reglas_riesgo']).sum()):,}")

    st.subheader("Priorización por reglas")
    st.caption(
        "Combina tres señales objetivas: valor en el 5% más alto, modalidad de "
        "contratación directa (menor competencia) y adiciones de plazo extensas. "
        "Más señales encendidas = mayor prioridad de revisión."
    )
    reglas = (f[f["senales"] >= 2]
              .sort_values(["senales", "Valor del Contrato"], ascending=False))
    vista_r = reglas[["Referencia del Contrato", "Nombre Entidad", "Valor del Contrato",
                      "Dias adicionados", "Modalidad de Contratacion", "Riesgo"]].copy()
    vista_r["Valor del Contrato"] = vista_r["Valor del Contrato"].apply(fmt_cop)
    vista_r["Dias adicionados"] = ("+" + vista_r["Dias adicionados"].fillna(0).astype(int).astype(str) + " d")
    st.dataframe(vista_r.head(50), width='stretch', hide_index=True)
    st.caption(f"{len(reglas):,} contratos con 2+ señales · "
               f"{int((f['senales'] == 3).sum()):,} críticos (3 señales) en la vista.")

    st.markdown("---")
    st.subheader("Detección de anomalías (Isolation Forest)")
    st.caption(
        "Un modelo de aprendizaje no supervisado que marca contratos atípicos "
        "combinando valor, duración, adiciones y modalidad, sin reglas predefinidas."
    )
    anom = f[f["es_anomalo"]].sort_values("anomaly_score")
    vista_a = anom[["Referencia del Contrato", "Nombre Entidad", "Valor del Contrato",
                    "Dias adicionados", "Modalidad de Contratacion", "anomaly_score"]].copy()
    vista_a["Valor del Contrato"] = vista_a["Valor del Contrato"].apply(fmt_cop)
    vista_a["Dias adicionados"] = vista_a["Dias adicionados"].fillna(0).astype(int)
    vista_a = vista_a.rename(columns={"anomaly_score": "Atipicidad (más bajo = más atípico)"})
    vista_a["Atipicidad (más bajo = más atípico)"] = vista_a["Atipicidad (más bajo = más atípico)"].round(3)
    st.dataframe(vista_a.head(50), width='stretch', hide_index=True)

    # Contraste modelo vs reglas
    st.markdown("---")
    st.subheader("¿Coinciden el modelo y las reglas?")
    ambos = int((f["es_anomalo"] & f["reglas_riesgo"]).sum())
    solo_m = int((f["es_anomalo"] & ~f["reglas_riesgo"]).sum())
    solo_r = int((~f["es_anomalo"] & f["reglas_riesgo"]).sum())
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Detectados por ambos", f"{ambos:,}", help="Se validan mutuamente")
    cc2.metric("Solo por el modelo", f"{solo_m:,}", help="Hallazgos que las reglas no vieron")
    cc3.metric("Solo por las reglas", f"{solo_r:,}", help="El modelo no los marcó")

    # Descarga
    st.markdown("---")
    export = f[f["reglas_riesgo"] | f["es_anomalo"]][
        ["Referencia del Contrato", "Nombre Entidad", "Valor del Contrato",
         "Dias adicionados", "Modalidad de Contratacion", "Riesgo",
         "es_anomalo", "anomaly_score"]
    ]
    st.download_button(
        "⬇️  Descargar contratos priorizados (CSV)",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name="contratos_priorizados_cartagena.csv",
        mime="text/csv",
    )

# ======================== METODOLOGÍA CIUDADANA ============================ #
with tab_metodo:
    st.subheader("¿Qué estás viendo y de dónde sale?")
    st.markdown(
        """
Este observatorio parte de los **datos abiertos** del **SECOP II**, la plataforma
donde las entidades públicas registran su contratación (administrada por Colombia
Compra Eficiente). Tomamos los contratos del Distrito de Cartagena, los **limpiamos**
y los presentamos aquí de forma navegable.

**Cómo priorizamos contratos.** Usamos dos enfoques que se complementan:

- **Reglas transparentes.** Marcamos un contrato cuando cumple señales objetivas:
  está entre el 5% de mayor valor, se hizo por **contratación directa** (la modalidad
  con menos competencia), o tiene **adiciones de plazo** inusualmente largas. Los cortes
  no son arbitrarios: salen de los propios datos (percentiles).

- **Un modelo que aprende solo (Isolation Forest).** Sin decirle qué buscar, el modelo
  detecta contratos que se salen del patrón general combinando varias características a la
  vez. Sirve para encontrar casos raros que una sola regla no vería.

Cuando **ambos** métodos coinciden en un contrato, la señal es más fuerte. Cuando difieren,
cada uno aporta una mirada distinta.
        """
    )
    st.markdown(
        '<div class="aviso"><b>Un contrato señalado no es un contrato corrupto.</b> '
        'Ser “atípico” significa que se aparta de lo común y por eso <b>merece revisión</b>. '
        'Puede tener una explicación legítima. Esta herramienta orienta el control ciudadano; '
        'no emite juicios ni acusaciones.</div>',
        unsafe_allow_html=True,
    )
    st.caption("Fuente de datos: SECOP II · Colombia Compra Eficiente (datos abiertos).")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Observatorio ciudadano de contratación pública de Cartagena · base metodológica. "
    "Datos: SECOP II (datos abiertos). Las anomalías son estadísticas, no determinaciones legales."
)




#Optimizado