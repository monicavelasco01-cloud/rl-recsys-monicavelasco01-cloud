"""Reto de la sesion 6 · Gane a dos lineas de codigo.

═══════════════════════════════════════════════════════════════════════════
QUE PASA
═══════════════════════════════════════════════════════════════════════════

Sobre los datos de ASSISTments, con particion TEMPORAL, esto es lo que sale:

    recomendador                 Recall@10   nDCG@10   cobertura
    repetir lo propio               0,7009    0,5880       0,973
    popularidad                     0,4121    0,2738       0,090
    kNN por items                   0,2613    0,1857       0,856
    al azar                         0,0990    0,0683       1,000
    novedad pura                    0,0949    0,0766       0,649

Lea la primera fila y la tercera. **El filtro colaborativo pierde contra una
linea base de dos lineas**, y por casi tres veces. Y «repetir lo propio» es
literalmente esto:

    los items que este usuario ya practico, el mas repetido primero,
    y despues los mas populares para rellenar

Lea tambien la ultima fila. «Novedad pura», que solo recomienda cosas que el
usuario NO ha hecho, saca menos que recomendar AL AZAR. La intuicion de que un
recomendador no debe repetir es, en estos datos, el error mas caro posible.

═══════════════════════════════════════════════════════════════════════════
QUE HAY QUE HACER
═══════════════════════════════════════════════════════════════════════════

Dos cosas, y la primera se entrega aunque la segunda no salga.

1. **El diagnostico, escrito en su bitacora antes de tocar el codigo.** Por que
   gana repetir. Que dice eso sobre estos datos. Y una prediccion: cuanto cree
   que va a sacar su recomendador.

2. **El recomendador.** Rellene ``mi_recomendador`` para superar a «repetir lo
   propio» sin encoger el catalogo. Tiene todo ``rlrs.recomendacion``
   disponible y puede escribir el suyo desde cero.

    uv run python scripts/reto6.py        mide y da el veredicto
    uv run pytest tests/test_reto6.py     comprueba el contrato

═══════════════════════════════════════════════════════════════════════════

Aviso: hay una forma facil de subir el Recall que consiste en recomendarle a
todo el mundo lo mismo. Por eso hay un criterio de cobertura, y por eso el
arnes le ensena las cuatro cifras y no una.
"""

from __future__ import annotations

from rlrs.recomendacion import (  # noqa: F401
    Particion,
    factorizacion_implicita,
    knn_items,
    por_popularidad,
    repetir_lo_propio,
)


def mi_recomendador(particion: Particion):
    """Devuelve una funcion que recomienda. **Esto es lo que usted escribe.**

    Parameters
    ----------
    particion:
        Tiene ``train`` (la lista de historiales que SI se pueden mirar),
        ``n_items`` y ``nombre``. **No tiene el conjunto de prueba**, y esa
        ausencia es lo unico que impide hacer trampa sin querer.

    Returns
    -------
    callable
        Recibe el historial de un usuario, como arreglo de items, y devuelve
        una lista de items ordenada de mejor a peor.
    """
    # ── su respuesta va aqui ──────────────────────────────────────────────
    return knn_items(particion, vecinos=20)
