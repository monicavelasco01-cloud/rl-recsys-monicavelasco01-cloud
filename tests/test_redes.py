"""Pruebas de la red escrita a mano.

La primera es la que de verdad importa: compara el gradiente calculado a mano
con el numerico. Si alguna vez se toca ``Capa.atras``, esa prueba es la que
avisa de que se rompio algo, y ninguna de las demas lo notaria.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrs.redes import Adam, Red, gradiente_numerico


def test_el_gradiente_a_mano_coincide_con_el_numerico():
    rng = np.random.default_rng(0)
    red = Red((4, 8, 3), seed=0)
    x = rng.normal(size=(5, 4))
    objetivo = rng.normal(size=(5, 3))

    def perdida():
        return float(((red.adelante(x) - objetivo) ** 2).mean())

    y = red.adelante(x)
    red.atras(2.0 * (y - objetivo) / y.size)
    analiticos = [g.copy() for g in red.gradientes()]

    for p, ga in zip(red.parametros(), analiticos):
        assert np.allclose(ga, gradiente_numerico(perdida, p), atol=1e-6)


def test_adam_baja_la_perdida():
    rng = np.random.default_rng(1)
    red = Red((4, 8, 3), seed=1)
    x = rng.normal(size=(6, 4))
    objetivo = rng.normal(size=(6, 3))
    opt = Adam(red.parametros(), lr=0.01)
    antes = float(((red.adelante(x) - objetivo) ** 2).mean())
    for _ in range(200):
        y = red.adelante(x)
        red.atras(2.0 * (y - objetivo) / y.size)
        opt.paso(red.parametros(), red.gradientes())
    assert float(((red.adelante(x) - objetivo) ** 2).mean()) < antes / 100.0


def test_la_ultima_capa_no_lleva_activacion():
    """Un valor de accion puede ser negativo. Si la red no puede producirlo,
    el error es silencioso: sigue entrenando y aprende peor."""
    red = Red((3, 6, 2), seed=0)
    red.capas[-1].b[:] = -5.0
    assert (red.adelante(np.zeros(3)) < 0).all()


def test_copiar_de_deja_las_dos_redes_iguales():
    a, b = Red((3, 5, 2), seed=0), Red((3, 5, 2), seed=1)
    x = np.ones((1, 3))
    assert not np.allclose(a.adelante(x), b.adelante(x))
    b.copiar_de(a)
    assert np.allclose(a.adelante(x), b.adelante(x))


def test_copiar_de_exige_la_misma_forma():
    with pytest.raises(ValueError, match="misma forma"):
        Red((3, 5, 2)).copiar_de(Red((3, 6, 2)))


def test_la_misma_semilla_da_la_misma_red():
    x = np.ones((1, 4))
    assert np.allclose(Red((4, 7, 2), seed=3).adelante(x), Red((4, 7, 2), seed=3).adelante(x))


def test_una_red_necesita_al_menos_dos_capas():
    with pytest.raises(ValueError):
        Red((5,))
