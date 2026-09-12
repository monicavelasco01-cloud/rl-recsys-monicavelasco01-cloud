"""Pruebas del contrato del reto 6.

Comprueban la forma de la respuesta y que no haya fuga, no la calidad: si el
recomendador gana lo decide `scripts/reto6.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from retos.reto6 import mi_recomendador

from rlrs.recomendacion import cargar, knn_items, particion_temporal

DATOS = cargar()
PARTICION = particion_temporal(DATOS)


def _por_defecto() -> bool:
    """True mientras la respuesta sea la que venia de fabrica."""
    mio = mi_recomendador(PARTICION)
    base = knn_items(PARTICION, vecinos=20)
    h = PARTICION.train[0]
    return list(mio(h))[:20] == list(base(h))[:20]


pytestmark = pytest.mark.skipif(
    _por_defecto(),
    reason="El reto 6 todavia no esta contestado: modifica 'mi_recomendador' en retos/reto6.py",
)


def test_devuelve_una_lista_de_items():
    r = mi_recomendador(PARTICION)(PARTICION.train[0])
    assert len(r) > 0, "No recomendaste nada."
    assert all(0 <= int(i) < PARTICION.n_items for i in r), (
        f"Recomendaste un item fuera del rango 0..{PARTICION.n_items - 1}."
    )


def test_no_repite_items_en_la_misma_lista():
    r = [int(i) for i in mi_recomendador(PARTICION)(PARTICION.train[0])][:10]
    assert len(r) == len(set(r)), (
        f"Tu top 10 tiene items repetidos: {r}. Recomendar dos veces lo mismo "
        "en la misma lista infla el Recall sin recomendar nada nuevo."
    )


def test_devuelve_al_menos_diez():
    r = mi_recomendador(PARTICION)(PARTICION.train[0])
    assert len(r) >= 10, (
        f"Devolviste {len(r)} items y las metricas son @10. Con menos de diez "
        "estas renunciando a puntos sin ganar nada."
    )


def test_es_determinista():
    f = mi_recomendador(PARTICION)
    h = PARTICION.train[3]
    a = [int(i) for i in f(h)][:10]
    b = [int(i) for i in f(h)][:10]
    assert a == b, (
        "Dos llamadas con el mismo historial dieron listas distintas. Si hay "
        "azar dentro, fija la semilla: sin eso nada de lo que midas es reproducible."
    )


def test_usuarios_distintos_reciben_cosas_distintas():
    """El sintoma de un recomendador que en realidad no mira al usuario."""
    f = mi_recomendador(PARTICION)
    listas = [tuple(int(i) for i in f(PARTICION.train[u])[:10]) for u in range(0, 60, 6)]
    assert len(set(listas)) > 1, (
        "Los diez usuarios de la muestra recibieron exactamente la misma lista. "
        "Eso es una linea base de popularidad con otro nombre."
    )


def test_no_mira_el_conjunto_de_prueba():
    """Si el recomendador cambia al cambiar el test, esta mirando donde no debe."""
    from rlrs.recomendacion import Particion

    sucia = Particion(PARTICION.nombre, PARTICION.train,
                      [np.array([], dtype=int) for _ in PARTICION.test],
                      PARTICION.n_items)
    a = [int(i) for i in mi_recomendador(PARTICION)(PARTICION.train[1])][:10]
    b = [int(i) for i in mi_recomendador(sucia)(sucia.train[1])][:10]
    assert a == b, (
        "Tu recomendador da resultados distintos cuando se le vacia el conjunto "
        "de prueba, asi que lo esta mirando. Eso es fuga de datos: en produccion "
        "ese conjunto no existe."
    )
