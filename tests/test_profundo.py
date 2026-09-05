"""Pruebas de DQN y del gradiente de politica.

Son pruebas de que el mecanismo hace lo que dice, no de que el agente aprenda
bien: eso se mide en ``experiments/profundo.py`` y depende de la semilla.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrs.aprox import caracteristicas_posicion
from rlrs.dqn import Memoria, dqn
from rlrs.envs import GridWorld
from rlrs.pg import actor_critico, reinforce, softmax


def entorno() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


# ── memoria ────────────────────────────────────────────────────────────────

def test_la_memoria_se_llena_y_da_la_vuelta():
    m = Memoria(capacidad=3, n_caracteristicas=2)
    for i in range(5):
        m.guardar(np.array([i, i]), i % 4, float(i), np.array([i, i]), False)
    assert len(m) == 3


def test_el_lote_sale_de_lo_guardado():
    m = Memoria(capacidad=10, n_caracteristicas=2)
    for i in range(4):
        m.guardar(np.array([i, 0.0]), 0, 1.0, np.array([0.0, 0.0]), False)
    s, a, r, s2, fin = m.lote(6, np.random.default_rng(0))
    assert s.shape == (6, 2)
    assert set(np.unique(s[:, 0])).issubset({0.0, 1.0, 2.0, 3.0})


# ── DQN ────────────────────────────────────────────────────────────────────

def test_dqn_devuelve_una_politica_de_una_accion_por_estado():
    env = entorno()
    ap = dqn(env, caracteristicas_posicion(env), episodes=20, seed=0)
    pol = ap.politica(entorno(), caracteristicas_posicion(entorno()))
    assert pol.shape == (env.n_states,)
    assert pol.min() >= 0 and pol.max() < env.n_actions


def test_dqn_reproduce_con_la_misma_semilla():
    env = entorno()
    a = dqn(entorno(), caracteristicas_posicion(env), episodes=20, seed=5)
    b = dqn(entorno(), caracteristicas_posicion(env), episodes=20, seed=5)
    assert np.allclose(a.red.parametros()[0], b.red.parametros()[0])


@pytest.mark.parametrize("memoria,objetivo", [(True, True), (True, False), (False, True), (False, False)])
def test_las_cuatro_combinaciones_de_la_ablacion_corren(memoria, objetivo):
    env = entorno()
    ap = dqn(env, caracteristicas_posicion(env), episodes=12, seed=0,
             usar_memoria=memoria, usar_red_objetivo=objetivo)
    assert np.isfinite(ap.normas).all()


def test_sin_red_objetivo_no_se_congela_nada():
    """Con red objetivo, los pesos de la copia solo cambian en los refrescos."""
    env = entorno()
    ap = dqn(env, caracteristicas_posicion(env), episodes=30, seed=0,
             refresco_objetivo=10_000, usar_red_objetivo=True)
    assert np.isfinite(ap.red.norma())


# ── gradiente de politica ──────────────────────────────────────────────────

def test_softmax_suma_uno():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_softmax_aguanta_numeros_grandes():
    """Sin restar el maximo, esto daria nan en silencio."""
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(), 1.0)


def test_reinforce_devuelve_una_politica_valida():
    env = entorno()
    ap = reinforce(env, caracteristicas_posicion(env), episodes=40, seed=0)
    pol = ap.politica(entorno(), caracteristicas_posicion(entorno()))
    assert pol.shape == (env.n_states,)


def test_actor_critico_corre_y_no_produce_nan():
    env = entorno()
    ap = actor_critico(env, caracteristicas_posicion(env), episodes=40, seed=0)
    assert np.isfinite(ap.retornos).all()


def test_la_entropia_mantiene_la_politica_menos_cerrada():
    """El termino de entropia existe para esto y se puede comprobar."""
    env = entorno()
    X = np.stack([caracteristicas_posicion(entorno())(s) for s in range(env.n_states)])
    sin = actor_critico(entorno(), caracteristicas_posicion(env), episodes=400, gamma=0.9,
                        seed=0, entropia=0.0)
    con = actor_critico(entorno(), caracteristicas_posicion(env), episodes=400, gamma=0.9,
                        seed=0, entropia=0.05)
    cerrada_sin = float(softmax(sin.red.adelante(X)).max(axis=1).mean())
    cerrada_con = float(softmax(con.red.adelante(X)).max(axis=1).mean())
    assert cerrada_con < cerrada_sin
