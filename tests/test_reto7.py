"""Pruebas de contrato del reto 7.

No comprueban que el recomendador sea bueno: eso lo mide `scripts/reto7.py`.
Comprueban que **es un recomendador**, que no hace trampa y que responde a un
usuario que no ha visto nunca, que es de lo que trata el reto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from retos.reto7 import mi_recomendador

from rlrs.recomendacion import (
    Particion,
    cargar,
    evaluar,
    particion_temporal,
    usuarios_nuevos,
)

K = 10


@pytest.fixture(scope="module")
def particiones():
    datos = cargar()
    normal = particion_temporal(datos)
    return normal, usuarios_nuevos(normal, fraccion=0.2, seed=0)


def test_devuelve_una_lista_de_enteros(particiones):
    normal, _ = particiones
    rec = mi_recomendador(normal)
    salida = rec(normal.train[0])
    assert isinstance(salida, (list, tuple, np.ndarray))
    assert all(isinstance(int(i), int) for i in salida)


def test_los_items_estan_en_el_catalogo(particiones):
    normal, _ = particiones
    rec = mi_recomendador(normal)
    for h in normal.train[:20]:
        for i in rec(h)[:K]:
            assert 0 <= int(i) < normal.n_items


def test_no_repite_items_en_la_misma_lista(particiones):
    normal, _ = particiones
    rec = mi_recomendador(normal)
    for h in normal.train[:20]:
        top = [int(i) for i in rec(h)][:K]
        assert len(top) == len(set(top)), "una lista con el mismo ítem dos veces"


def test_devuelve_al_menos_k_items(particiones):
    """Un recomendador que devuelve tres cosas cuando caben diez esta perdiendo Recall."""
    normal, _ = particiones
    rec = mi_recomendador(normal)
    cortos = sum(1 for h in normal.train[:50] if len([int(i) for i in rec(h)]) < K)
    assert cortos == 0, f"{cortos} de 50 usuarios reciben menos de {K} ítems"


def test_no_mira_el_conjunto_de_prueba(particiones):
    """Se le cambia el test por ruido: si el resultado cambia, lo estaba mirando."""
    normal, _ = particiones
    rng = np.random.default_rng(0)
    falsa = Particion(
        nombre="con el test cambiado",
        train=normal.train,
        test=[rng.integers(0, normal.n_items, len(t)) for t in normal.test],
        n_items=normal.n_items,
    )
    a = evaluar(normal, mi_recomendador(normal), k=K)
    rec_b = mi_recomendador(falsa)
    b = [int(i) for i in rec_b(normal.train[0])][:K]
    c = [int(i) for i in mi_recomendador(normal)(normal.train[0])][:K]
    assert b == c, "el recomendador cambia cuando cambia el test: lo está mirando"
    assert a.recall >= 0.0


def test_responde_a_un_usuario_nuevo(particiones):
    """El corazon del reto: no puede devolver una lista vacia.

    Esta prueba no exige acertar. Exige **responder**. Un recomendador que ante
    un usuario desconocido devuelve una lista vacia no es un recomendador malo:
    es un recomendador que no existe para ese usuario.
    """
    _, frio = particiones
    rec = mi_recomendador(frio)
    vacias = sum(1 for h in frio.train[:100] if len([int(i) for i in rec(h)]) == 0)
    assert vacias == 0, (
        f"{vacias} de 100 usuarios nuevos reciben una lista vacía. "
        "Ese es el reto, y por defecto está puesto para que falle.")
