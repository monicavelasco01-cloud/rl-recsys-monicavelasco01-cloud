"""Pruebas de la iteracion de valor.

La prueba mas importante de todas es ``test_la_politica_optima_evita_la_trampa``:
comprueba una propiedad del *comportamiento*, no un numero. Ese es el tipo de
prueba que hay que escribir en aprendizaje por refuerzo.
"""

import numpy as np
import pytest

from rlrs.dp import value_iteration
from rlrs.envs import GridWorld


def test_converge_y_devuelve_las_formas_correctas():
    env = GridWorld()
    values, policy, sweeps = value_iteration(env, gamma=0.9)
    assert values.shape == (env.n_states,)
    assert policy.shape == (env.n_states,)
    assert 0 < sweeps < 1000


def test_todos_los_valores_son_positivos_con_gamma_alto():
    """Con gamma = 0.9 compensa emprender el viaje desde cualquier casilla."""
    env = GridWorld()
    values, _, _ = value_iteration(env, gamma=0.9)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.is_wall((r, c)) or env.is_terminal((r, c)):
                continue
            assert values[env.state_index((r, c))] > 0


def test_un_gamma_bajo_vuelve_miope_al_agente():
    """Con gamma = 0.55 el descuento se come la meta y la fila lejana se hace negativa."""
    env = GridWorld()
    values, _, _ = value_iteration(env, gamma=0.55)
    assert values[env.state_index((3, 0))] < 0


def test_el_valor_crece_al_acercarse_a_la_meta():
    env = GridWorld()
    values, _, _ = value_iteration(env, gamma=0.9)
    cerca = values[env.state_index((0, 3))]
    lejos = values[env.state_index((3, 0))]
    assert cerca > lejos


def test_la_politica_optima_evita_la_trampa():
    """Con mucho ruido, la casilla bajo la trampa deja de apuntar hacia ella."""
    from rlrs.envs import UP

    env = GridWorld(noise=0.5)
    _, policy, _ = value_iteration(env, gamma=0.9)
    assert policy[env.state_index((1, 3))] != UP


def test_sin_ruido_la_ruta_es_directa():
    from rlrs.envs import RIGHT

    env = GridWorld(noise=0.0)
    _, policy, _ = value_iteration(env, gamma=0.9)
    assert policy[env.state_index((0, 3))] == RIGHT


def test_gamma_mas_alto_necesita_mas_barridos():
    env = GridWorld()
    _, _, pocos = value_iteration(env, gamma=0.5)
    _, _, muchos = value_iteration(env, gamma=0.95)
    assert muchos > pocos


def test_gamma_invalido_se_rechaza():
    env = GridWorld()
    with pytest.raises(ValueError):
        value_iteration(env, gamma=1.0)


def test_es_reproducible():
    env = GridWorld()
    a, _, _ = value_iteration(env, gamma=0.9)
    b, _, _ = value_iteration(env, gamma=0.9)
    assert np.allclose(a, b)
