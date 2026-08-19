# -*- coding: utf-8 -*-
"""
Lecturas en lenguaje llano de lo que muestra cada apartado del observatorio.

La persona que entra aquí no tiene por qué saber qué es una mediana, un percentil
ni una modalidad de contratación. Cada función de este módulo mira los contratos
que el usuario tiene delante —los filtrados, no el universo— y devuelve tres
frases: qué es lo que está viendo, qué dicen sus datos concretos, y qué pregunta
vale la pena hacerse a partir de ahí.

Son funciones puras sobre el DataFrame para poder probarlas: si una lectura
afirma "el 43% del dinero", esa cifra sale de los datos y no de un texto fijo que
alguien olvidó actualizar.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline import fmt_cop, fmt_entero

VACIO = "No hay contratos en esta vista, así que no hay nada que interpretar todavía."


@dataclass(frozen=True)
class Lectura:
    """Explicación de un apartado, en tres pasos."""

    que_es: str
    que_dice: str
    por_que_importa: str


def _pct(parte, total) -> str:
    """
    Porcentaje redondeado, tolerante con el total en cero.

    Una cantidad pequeña pero real no puede mostrarse como "0%": el lector
    entendería que no hay ninguno. Por debajo del 1% se dice así.
    """
    if not total:
        return "—"
    p = parte / total * 100
    if 0 < p < 1:
        return "menos del 1%"
    return f"{p:.0f}%"


# Mismo formato de miles que usa el resto de la aplicación.
_miles = fmt_entero


def para_markdown(texto: str) -> str:
    """
    Escapa el signo de peso antes de pintar un texto como Markdown.

    Streamlit lee `$…$` como una fórmula LaTeX, así que una frase con dos
    importes —«vale $137 M, y el típico $20 M»— se renderiza como matemáticas
    ilegibles en vez de como dinero. Las lecturas se guardan sin escapar, para
    que sirvan igual a una interfaz que no use Markdown; el escape se aplica
    solo en el momento de pintarlas.
    """
    return str(texto).replace("$", "\\$")


def _nombre(x, largo: int = 60) -> str:
    txt = str(x).strip()
    return txt if len(txt) <= largo else txt[: largo - 1].rstrip() + "…"


def _vacia(que_es: str) -> Lectura:
    return Lectura(que_es=que_es, que_dice=VACIO, por_que_importa="")


# --------------------------------------------------------------------------- #
# Resumen de la vista completa
# --------------------------------------------------------------------------- #
def resumen_vista(f: pd.DataFrame) -> list[str]:
    """
    El párrafo que debería bastar si alguien solo lee una cosa: qué hay en esta
    vista, quién maneja el dinero, cómo se adjudicó y qué conviene revisar.
    """
    if len(f) == 0:
        return [VACIO]

    puntos = []
    total = f["Valor del Contrato"].sum()
    n = f["ID Contrato"].nunique()
    puntos.append(
        f"Estás viendo **{_miles(n)} contratos** que suman **{fmt_cop(total)}** de dinero público."
    )

    por_entidad = f.groupby("Nombre Entidad", observed=True)["Valor del Contrato"].sum()
    if len(por_entidad) and total:
        primera = por_entidad.sort_values(ascending=False)
        puntos.append(
            f"La entidad que más contrata es **{_nombre(primera.index[0])}**, "
            f"con el **{_pct(primera.iloc[0], total)}** de todo ese dinero."
        )

    directa = f["es_directa"].mean()
    if pd.notna(directa):
        puntos.append(
            f"El **{directa * 100:.0f}%** de estos contratos se adjudicó por "
            "**contratación directa**, es decir, sin abrir una competencia entre empresas."
        )

    prov = f.groupby("Proveedor Adjudicado", observed=True)["Valor del Contrato"].sum()
    if len(prov) > 6 and total:
        puntos.append(
            f"De **{_miles(len(prov))} empresas** contratadas, las **6 más grandes** se llevan el "
            f"**{_pct(prov.sort_values(ascending=False).head(6).sum(), total)}** del dinero."
        )

    n_reglas = int(f["reglas_riesgo"].sum())
    n_anom = int(f["es_anomalo"].sum())
    puntos.append(
        f"**{_miles(n_reglas)} contratos** acumulan dos o más señales de alerta y **{_miles(n_anom)}** "
        "se salen del patrón según el modelo. Son los que conviene mirar primero, "
        "y verlos aquí no significa que tengan nada malo."
    )
    return puntos


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def lectura_kpis(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Seis cifras que resumen los contratos que estás viendo. Cambian cuando "
        "mueves los filtros de la izquierda: son el retrato de tu selección, no de "
        "toda la ciudad."
    )
    if len(f) == 0:
        return _vacia(que_es)

    promedio = f["Valor del Contrato"].mean()
    mediana = f["Valor del Contrato"].median()
    dice = (
        f"El contrato promedio de esta vista vale **{fmt_cop(promedio)}**, pero el contrato "
        f"típico —el del medio— vale **{fmt_cop(mediana)}**."
    )
    if pd.notna(promedio) and pd.notna(mediana) and mediana > 0 and promedio > 2 * mediana:
        dice += (
            " Esa diferencia tan grande significa que unos pocos contratos enormes están "
            "inflando el promedio: la mayoría son mucho más pequeños de lo que sugiere."
        )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Cuando alguien cita “el contrato promedio”, conviene preguntar por la mediana. "
            "El promedio se deja arrastrar por los casos extremos; la mediana describe mejor "
            "lo que pasa habitualmente."
        ),
    )


# --------------------------------------------------------------------------- #
# Panorama del gasto
# --------------------------------------------------------------------------- #
def lectura_entidades(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Las seis entidades públicas que más dinero contrataron, sumando todos sus "
        "contratos. Una “entidad” es el organismo que firma y paga: la Alcaldía, un "
        "hospital público, el SENA, un instituto distrital."
    )
    if len(f) == 0:
        return _vacia(que_es)

    por_entidad = (
        f.groupby("Nombre Entidad", observed=True)["Valor del Contrato"]
        .sum().sort_values(ascending=False)
    )
    total = f["Valor del Contrato"].sum()
    if por_entidad.empty or not total:
        return _vacia(que_es)

    dice = (
        f"**{_nombre(por_entidad.index[0])}** encabeza la lista con "
        f"**{fmt_cop(por_entidad.iloc[0])}**, el **{_pct(por_entidad.iloc[0], total)}** del "
        f"dinero de esta vista."
    )
    if len(por_entidad) > 1:
        dice += (
            f" Le sigue **{_nombre(por_entidad.index[1])}** con "
            f"**{_pct(por_entidad.iloc[1], total)}**."
        )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Saber quién maneja el presupuesto es el primer paso para pedir cuentas: "
            "los derechos de petición y las solicitudes de información se dirigen a la "
            "entidad que firmó el contrato, no a la empresa que lo recibió."
        ),
    )


def lectura_modalidades(f: pd.DataFrame) -> Lectura:
    que_es = (
        "La **modalidad** es la forma en que se eligió a la empresa. En una **licitación** "
        "o un **concurso** varias empresas compiten y gana una. En la **contratación "
        "directa** la entidad escoge a la empresa sin abrir esa competencia."
    )
    if len(f) == 0:
        return _vacia(que_es)

    conteo = f["Modalidad de Contratacion"].value_counts()
    conteo = conteo[conteo > 0]
    if conteo.empty:
        return _vacia(que_es)

    directa = f["es_directa"].mean()
    dice = (
        f"La modalidad más usada es **{_nombre(conteo.index[0], 45)}** "
        f"({_pct(conteo.iloc[0], len(f))} de los contratos). En total, el "
        f"**{directa * 100:.0f}%** de esta vista se adjudicó de forma directa."
    )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "La ley permite contratar directamente en casos concretos: urgencias, servicios "
            "profesionales, cuando solo existe un proveedor posible. No es ilegal ni es, por "
            "sí solo, indicio de nada. Pero al no haber competencia nadie ofreció un precio "
            "mejor, así que es más difícil comprobar que se pagó lo justo. Por eso vale la "
            "pena vigilarlo, sobre todo cuando se vuelve la regla y no la excepción."
        ),
    )


def lectura_evolucion(f: pd.DataFrame) -> Lectura:
    que_es = "Cuánto dinero se contrató cada año, según la fecha de firma del contrato."
    if len(f) == 0:
        return _vacia(que_es)

    por_anio = f.dropna(subset=["Fecha de Firma"]).groupby("anio_firma")["Valor del Contrato"].sum()
    por_anio = por_anio[por_anio > 0]
    if por_anio.empty:
        return _vacia(que_es)

    pico = por_anio.idxmax()
    dice = f"El año de mayor contratación fue **{int(pico)}**, con **{fmt_cop(por_anio.max())}**."
    if len(por_anio) >= 2:
        ordenado = por_anio.sort_index()
        ultimo, previo = ordenado.iloc[-1], ordenado.iloc[-2]
        if previo:
            cambio = (ultimo - previo) / previo * 100
            comparativo = "más" if cambio >= 0 else "menos"
            dice += (
                f" En **{int(ordenado.index[-1])}** se contrataron **{fmt_cop(ultimo)}**, "
                f"un **{abs(cambio):.0f}% {comparativo}** que el año anterior."
            )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Los saltos grandes de un año a otro tienen explicación casi siempre: un cambio "
            "de gobierno, una emergencia, una obra grande que arranca. Averiguar cuál fue la "
            "razón es una buena pregunta para la entidad. Ten en cuenta que el año en curso "
            "aparecerá incompleto, porque todavía no ha terminado."
        ),
    )


def lectura_valor_mediano(f: pd.DataFrame) -> Lectura:
    que_es = (
        "El valor **del medio** de los contratos en cada modalidad: la mitad cuesta menos "
        "y la mitad cuesta más. Se usa la mediana en vez del promedio porque un solo "
        "contrato gigantesco distorsionaría el promedio de toda la modalidad. Solo "
        "aparecen las modalidades con al menos 30 contratos, para no sacar conclusiones "
        "de dos o tres casos."
    )
    if len(f) == 0:
        return _vacia(que_es)

    resumen = (
        f.groupby("Modalidad de Contratacion", observed=True)["Valor del Contrato"]
        .agg(["median", "count"])
    )
    resumen = resumen[resumen["count"] >= 30].sort_values("median", ascending=False)
    if resumen.empty:
        return Lectura(
            que_es=que_es,
            que_dice=(
                "Ninguna modalidad llega a 30 contratos en esta vista, así que no se muestra "
                "ninguna: con tan pocos casos, la cifra diría más del azar que de la realidad."
            ),
            por_que_importa="Prueba a ampliar los filtros para ver la comparación.",
        )

    dice = (
        f"Los contratos más grandes son los de **{_nombre(resumen.index[0], 45)}**: "
        f"la mitad supera los **{fmt_cop(resumen['median'].iloc[0])}**."
    )
    if len(resumen) > 1:
        dice += (
            f" En el otro extremo, **{_nombre(resumen.index[-1], 45)}** tiene una mediana de "
            f"**{fmt_cop(resumen['median'].iloc[-1])}**."
        )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Compara esta gráfica con la de modalidades: si los contratos más caros se "
            "concentran justo en las modalidades donde no hubo competencia, ahí hay una "
            "pregunta que vale la pena hacer."
        ),
    )


def lectura_pyme(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Qué porcentaje de los contratos se lleva una **Pyme** (pequeña o mediana empresa) "
        "según el tamaño del contrato. Cada barra es un rango de valor: los contratos "
        "pequeños a la izquierda, los más grandes a la derecha."
    )
    if len(f) == 0:
        return _vacia(que_es)

    tabla = f.groupby("Rango Valor", observed=True)["Es Pyme"].mean().dropna() * 100
    if tabla.empty:
        return _vacia(que_es)

    dice = f"En conjunto, el **{f['Es Pyme'].mean() * 100:.0f}%** de los contratos fue a una Pyme."
    if len(tabla) >= 2:
        # No se asume que la participación caiga a medida que crece el contrato:
        # se dice dónde es mayor y dónde menor, sea cual sea la forma real.
        dice += (
            f" Su presencia es mayor en los contratos de **{tabla.idxmax()}** "
            f"(**{tabla.max():.0f}%**) y menor en los de **{tabla.idxmin()}** "
            f"(**{tabla.min():.0f}%**)."
        )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "El dinero público también mueve la economía local. Esta gráfica permite "
            "comprobar si las empresas pequeñas llegan a contratos de todos los tamaños o "
            "se quedan concentradas en unos pocos. Si quedan fuera de algún rango, conviene "
            "preguntar si los requisitos exigidos allí las están excluyendo sin una razón "
            "de peso."
        ),
    )


def lectura_pareto(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Cada barra azul es un proveedor —la empresa o persona que recibió el contrato— y "
        "muestra qué porcentaje del dinero total se llevó. La última, «Otros», junta a todos "
        "los demás. La línea naranja va sumando esos porcentajes: si sube de golpe al "
        "principio, unas pocas empresas acaparan casi todo; si sube despacio y la barra de "
        "«Otros» es la más alta, el dinero está repartido entre muchísimas."
    )
    if len(f) == 0:
        return _vacia(que_es)

    prov = (
        f.groupby("Proveedor Adjudicado", observed=True)["Valor del Contrato"]
        .sum().sort_values(ascending=False)
    )
    total = prov.sum()
    if prov.empty or not total:
        return _vacia(que_es)

    top6 = prov.head(6).sum()
    dice = (
        f"Se contrataron **{_miles(len(prov))} proveedores distintos**. Los **6 mayores** "
        f"concentran el **{_pct(top6, total)}** del dinero, y el primero de todos "
        f"—**{_nombre(prov.index[0], 45)}**— se lleva por sí solo el "
        f"**{_pct(prov.iloc[0], total)}**."
    )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Que pocos proveedores concentren mucho dinero no es ilegal: puede que sean los "
            "únicos capaces de hacer esa obra, o que se trate de un consorcio creado para un "
            "proyecto grande. Pero es la señal más habitual para revisar si hubo competencia "
            "real, y se vuelve más relevante si además esos contratos fueron directos."
        ),
    )


# --------------------------------------------------------------------------- #
# Contratos priorizados
# --------------------------------------------------------------------------- #
def _frases_umbrales(uv, ud, udir) -> list[str]:
    """Una frase por cada umbral que se haya podido calcular."""
    frases = []
    if pd.notna(uv):
        frases.append(
            f"Un contrato se considera **caro** si supera **{fmt_cop(uv)}**: "
            "solo 5 de cada 100 llegan ahí."
        )
    if pd.notna(ud):
        frases.append(
            "Se considera que tiene **adiciones largas** si le sumaron más de "
            f"**{_miles(ud)} días** de plazo después de firmarlo."
        )
    if pd.notna(udir):
        frases.append(
            "Una entidad se considera **intensiva en contratación directa** si la usa "
            f"en más del **{udir * 100:.0f}%** de sus contratos."
        )
    return frases


def lectura_umbrales(data: pd.DataFrame) -> Lectura:
    a = data.attrs
    uv, ud, udir = a.get("umbral_valor"), a.get("umbral_dias"), a.get("umbral_directa")
    return Lectura(
        que_es=(
            "Los tres cortes a partir de los cuales un contrato empieza a llamar la atención. "
            "No los elegimos a dedo: salen de comparar cada contrato con todos los demás."
        ),
        # Cada umbral se explica por separado: si a los datos les falta uno
        # —por ejemplo, ningún contrato tiene adiciones— el lector debe seguir
        # viendo la explicación de los que sí existen.
        que_dice=" ".join(_frases_umbrales(uv, ud, udir))
        or "Todavía no hay datos suficientes para calcular los umbrales.",
        por_que_importa=(
            "Al salir de los propios datos, los cortes se ajustan solos: si mañana cambia la "
            "contratación de la ciudad, cambian los umbrales. Nadie puede acomodarlos para "
            "que un contrato concreto quede dentro o fuera."
        ),
    )


def lectura_reglas(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Una lista de contratos ordenada por cuántas señales de alerta acumulan. Las señales "
        "son tres: que el contrato esté entre los más caros, que lo firme una entidad que "
        "recurre a la contratación directa más que las demás, y que le hayan sumado mucho "
        "plazo después de firmarlo. Aquí solo aparecen los que encienden **dos o más**."
    )
    if len(f) == 0:
        return _vacia(que_es)

    n2 = int((f["senales"] >= 2).sum())
    n3 = int((f["senales"] == 3).sum())
    if n2 == 0:
        return Lectura(
            que_es=que_es,
            que_dice=(
                "Ningún contrato de esta vista acumula dos o más señales. Es una buena "
                "noticia, no un error."
            ),
            por_que_importa=(
                "Puedes ampliar los filtros para revisar un conjunto más grande de contratos."
            ),
        )

    dice = (
        f"**{_miles(n2)} contratos** de esta vista encienden dos o más señales "
        f"({_pct(n2, len(f))} del total). De ellos, **{_miles(n3)}** encienden las tres a la vez."
    )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Esta lista es un punto de partida para preguntar, no una acusación. Un contrato "
            "puede ser caro, directo y con plazo ampliado por razones perfectamente legítimas. "
            "Lo que dice la lista es: si vas a revisar algo con tiempo limitado, empieza por aquí."
        ),
    )


def lectura_anomalias(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Aquí no aplicamos reglas fijas: un programa revisa todos los contratos a la vez "
        "—cuánto valen, cuánto duran, cuánto plazo se les añadió y cómo se adjudicaron— y "
        "señala los que no se parecen a ninguno de los demás. Es como notar que una casa "
        "desentona en una calle sin haber definido antes cómo debe ser una casa."
    )
    if len(f) == 0:
        return _vacia(que_es)

    n = int(f["es_anomalo"].sum())
    if n == 0:
        return Lectura(
            que_es=que_es,
            que_dice="El modelo no encontró contratos fuera de patrón en esta vista.",
            por_que_importa="Prueba a ampliar los filtros para revisar más contratos.",
        )

    mediana_anom = f.loc[f["es_anomalo"], "Valor del Contrato"].median()
    mediana_resto = f.loc[~f["es_anomalo"], "Valor del Contrato"].median()
    dice = (
        f"El modelo marcó **{_miles(n)} contratos** ({_pct(n, len(f))} de esta vista). "
        f"El contrato marcado típico vale **{fmt_cop(mediana_anom)}**, frente a "
        f"**{fmt_cop(mediana_resto)}** del resto."
    )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "“Atípico” significa **distinto de lo habitual**, y nada más. Un contrato puede "
            "ser rarísimo y estar impecable: una obra única, una emergencia real. Lo que "
            "aporta el modelo es encontrar combinaciones extrañas que ninguna regla fija "
            "habría pensado en buscar."
        ),
    )


def lectura_coincidencia(f: pd.DataFrame) -> Lectura:
    que_es = (
        "Comparamos los dos métodos: las reglas, que sabemos exactamente qué buscan, y el "
        "modelo, que descubre patrones por su cuenta. Aquí se ve en cuántos contratos "
        "coinciden."
    )
    if len(f) == 0:
        return _vacia(que_es)

    ambos = int((f["es_anomalo"] & f["reglas_riesgo"]).sum())
    solo_m = int((f["es_anomalo"] & ~f["reglas_riesgo"]).sum())
    solo_r = int((~f["es_anomalo"] & f["reglas_riesgo"]).sum())
    n_reglas = ambos + solo_r
    if n_reglas == 0 and solo_m == 0:
        return Lectura(
            que_es=que_es,
            que_dice="Ningún método señaló contratos en esta vista.",
            por_que_importa="",
        )

    dice = (
        f"**{_miles(ambos)} contratos** los señalan los dos métodos a la vez. "
        f"**{_miles(solo_m)}** solo los ve el modelo y **{_miles(solo_r)}** solo las reglas."
    )
    if n_reglas:
        dice += (
            f" Dicho de otro modo: **{_pct(ambos, n_reglas)}** de los contratos que las "
            "reglas consideran prioritarios también le resultan extraños al modelo."
        )
    return Lectura(
        que_es=que_es,
        que_dice=dice,
        por_que_importa=(
            "Cuando dos métodos independientes coinciden en un contrato, la señal es más "
            "sólida: son los primeros que revisaría. Los que solo ve el modelo son el "
            "hallazgo más interesante, porque nadie le dijo qué buscar. Y los que solo ven "
            "las reglas son casos claros que el modelo, al mirar el conjunto, considera "
            "todavía dentro de lo normal."
        ),
    )


# --------------------------------------------------------------------------- #
# Glosario
# --------------------------------------------------------------------------- #
GLOSARIO: dict[str, str] = {
    "SECOP II": (
        "La plataforma del Estado donde las entidades públicas deben registrar sus "
        "contratos. Es la fuente de todo lo que ves aquí: no son datos filtrados ni "
        "obtenidos por vías indirectas, son públicos por ley."
    ),
    "Entidad": (
        "El organismo público que firma y paga el contrato: la Alcaldía, un hospital "
        "público, el SENA, un instituto distrital. Es a quien se le piden cuentas."
    ),
    "Proveedor": (
        "La empresa o persona que recibe el contrato y presta el servicio o ejecuta la obra."
    ),
    "Modalidad de contratación": (
        "La forma en que se eligió al proveedor. En una licitación varias empresas compiten; "
        "en la contratación directa la entidad escoge sin abrir competencia."
    ),
    "Contratación directa": (
        "La entidad elige a la empresa sin convocar a otras. La ley lo permite en casos "
        "concretos (urgencias, servicios profesionales, proveedor único). Al no haber "
        "competencia es más difícil comprobar que el precio fue el mejor posible."
    ),
    "Adición de plazo": (
        "Días que se le suman al contrato después de haberlo firmado. Puede ser normal "
        "—una obra que se retrasa por lluvias— o puede indicar que el plazo inicial no era "
        "realista."
    ),
    "Pyme": (
        "Pequeña o mediana empresa. Se marca en el contrato para poder medir si el gasto "
        "público llega también a las empresas locales más pequeñas."
    ),
    "Mediana": (
        "El valor del medio: la mitad de los contratos cuesta menos y la mitad cuesta más. "
        "Describe mejor “lo normal” que el promedio, al que un solo contrato gigante puede "
        "desviar por completo."
    ),
    "Percentil 95": (
        "El punto por encima del cual está solo el 5% más alto. Si un contrato supera el "
        "percentil 95 de valor, es de los 5 más caros de cada 100."
    ),
    "Contrato atípico": (
        "Un contrato que se aparta del patrón general. **No** quiere decir irregular ni "
        "corrupto: quiere decir distinto, y por eso merece una mirada."
    ),
    "Isolation Forest": (
        "El programa que busca los contratos atípicos. Va separando contratos por sus "
        "características al azar; los que quedan aislados en pocos pasos son los raros. "
        "Nadie le dice qué buscar: lo deduce del conjunto."
    ),
}


def glosario_ordenado() -> list[tuple[str, str]]:
    """Términos del glosario en el orden en que aparecen en la interfaz."""
    return list(GLOSARIO.items())


def terminos_presentes(texto: str) -> list[str]:
    """Términos del glosario mencionados en un texto (para pruebas de coherencia)."""
    bajo = texto.lower()
    return [t for t in GLOSARIO if t.lower() in bajo]


__all__ = [
    "Lectura",
    "GLOSARIO",
    "glosario_ordenado",
    "lectura_anomalias",
    "lectura_coincidencia",
    "lectura_entidades",
    "lectura_evolucion",
    "lectura_kpis",
    "lectura_modalidades",
    "lectura_pareto",
    "lectura_pyme",
    "lectura_reglas",
    "lectura_umbrales",
    "lectura_valor_mediano",
    "para_markdown",
    "resumen_vista",
    "terminos_presentes",
]
