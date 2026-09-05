"""Pruebas de la aproximacion lineal y de la triada mortal.

Hay dos que valen mas que las otras. Una comprueba que la tabla de siempre es un
caso particular de la aproximacion lineal, porque si eso no se cumple todo lo
demas del modulo esta mal. La otra comprueba la triada por eliminacion: con las
tres patas diverge, y quitando cualquiera de las tres converge.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrs.aprox import (
    caracteristicas_agregadas,
    caracteristicas_posicion,
    caracteristicas_tabulares,
    error_maximo,
    estrella_de_baird,
    sarsa_semigradiente,
    triada_completa,
)
from rlrs.dp import value_iteration
from rlrs.envs import GridWorld


def entorno() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


# ── caracteristicas ────────────────────────────────────────────────────────

def test_las_tabulares_encienden_una_sola_casilla():
    env = entorno()
    phi = caracteristicas_tabulares(env)
    x = phi(7)
    assert x.sum() == pytest.approx(1.0)
    assert x[7] == pytest.approx(1.0)


def test_las_de_posicion_son_cuatro_numeros():
    env = entorno()
    assert caracteristicas_posicion(env)(0).size == 4


def test_la_agregacion_obliga_a_compartir_valor():
    """Dos casillas del mismo bloque tienen exactamente la misma descripcion.

    Esa es la causa del desastre que se mide en la guia: si dos estados son
    indistinguibles para el aproximador, no hay pesos que puedan darles valores
    distintos, por mucho que se entrene.
    """
    env = entorno()
    phi = caracteristicas_agregadas(env, bloque=2)
    a = env.state_index((0, 0))
    b = env.state_index((0, 1))
    assert np.array_equal(phi(a), phi(b))


def test_estados_de_bloques_distintos_se_distinguen():
    env = entorno()
    phi = caracteristicas_agregadas(env, bloque=2)
    assert not np.array_equal(phi(env.state_index((0, 0))), phi(env.state_index((2, 2))))


# ── la tabla es un caso particular ─────────────────────────────────────────

def test_con_caracteristicas_tabulares_se_recupera_la_tabla():
    """El peso de la casilla ``s`` para la accion ``a`` **es** ``Q(s,a)``."""
    env = entorno()
    phi = caracteristicas_tabulares(env)
    ap = sarsa_semigradiente(env, phi, episodes=300, gamma=0.9, alpha=0.1, seed=0)
    tabla = ap.tabla_q(env, phi)
    assert np.allclose(tabla, ap.w)


def test_la_aproximacion_tabular_se_acerca_a_la_solucion_exacta():
    env = entorno()
    valores, _, _ = value_iteration(env, gamma=0.9)
    phi = caracteristicas_tabulares(env)
    ap = sarsa_semigradiente(env, phi, episodes=5000, gamma=0.9, alpha=0.05, seed=0)
    assert error_maximo(ap.tabla_q(env, phi), valores, env) < 0.5


def test_menos_pesos_significa_mas_error():
    """Generalizar cuesta precision. Aqui se cobra el precio, con numeros."""
    env = entorno()
    valores, _, _ = value_iteration(env, gamma=0.9)
    errores = []
    for fabrica in (caracteristicas_tabulares, caracteristicas_posicion):
        phi = fabrica(env)
        ap = sarsa_semigradiente(entorno(), phi, episodes=5000, gamma=0.9, alpha=0.05, seed=0)
        errores.append(error_maximo(ap.tabla_q(env, phi), valores, env))
    assert errores[0] < errores[1]


def test_reproduce_con_la_misma_semilla():
    env = entorno()
    phi = caracteristicas_posicion(env)
    a = sarsa_semigradiente(entorno(), phi, episodes=200, seed=3)
    b = sarsa_semigradiente(entorno(), phi, episodes=200, seed=3)
    assert np.allclose(a.w, b.w)


# ── la triada ──────────────────────────────────────────────────────────────

def test_con_las_tres_patas_diverge():
    d = estrella_de_baird(pasos=5000, alpha=0.01, seed=0)
    assert d.diverge
    assert d.norma[-1] > 1e4


@pytest.mark.parametrize(
    "cambio",
    [
        {"aproximacion": "tabular"},
        {"semigradiente": False},
        {"fuera_de_politica": False},
    ],
)
def test_quitando_cualquier_pata_converge(cambio):
    d = estrella_de_baird(pasos=5000, alpha=0.01, seed=0, **cambio)
    assert not d.diverge
    assert d.norma[-1] < 100.0


def test_la_triada_completa_devuelve_un_solo_divergente():
    resultados = triada_completa(pasos=5000, alpha=0.01, seed=0)
    assert sum(d.diverge for d in resultados) == 1
    assert resultados[0].diverge


def test_la_divergencia_no_depende_de_la_semilla():
    for semilla in (0, 1, 2, 3, 4):
        assert estrella_de_baird(pasos=5000, alpha=0.01, seed=semilla).diverge


def test_aproximacion_desconocida_falla():
    with pytest.raises(ValueError, match="aproximacion desconocida"):
        estrella_de_baird(pasos=10, aproximacion="polinomica")  # type: ignore[arg-type]
