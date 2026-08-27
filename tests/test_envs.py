"""Pruebas del entorno.

Estas pruebas existen por una razon concreta: en aprendizaje por refuerzo un
error no lanza una excepcion, produce una curva de aprendizaje plausible. La
unica defensa es comprobar el entorno por separado, antes de culpar al agente.
"""

import numpy as np
import pytest

from rlrs.envs import DOWN, LEFT, RIGHT, UP, GridWorld


def test_dimensiones_por_defecto():
    env = GridWorld()
    assert env.n_states == 20
    assert env.n_actions == 4


def test_indices_y_posiciones_son_inversos():
    env = GridWorld()
    for idx in range(env.n_states):
        assert env.state_index(env.state_pos(idx)) == idx


def test_las_probabilidades_de_transicion_suman_uno():
    env = GridWorld(noise=0.3)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.is_wall((r, c)) or env.is_terminal((r, c)):
                continue
            for a in range(env.n_actions):
                total = sum(p for _, p in env.transitions((r, c), a))
                assert total == pytest.approx(1.0)


def test_el_agente_no_atraviesa_muros_ni_bordes():
    env = GridWorld(noise=0.0)
    # Desde (0,0), ir hacia arriba choca con el borde y deja al agente donde estaba.
    assert env.transitions((0, 0), UP) == [((0, 0), 1.0)]
    # Desde (1,0), ir a la derecha choca con el muro de (1,1).
    assert env.transitions((1, 0), RIGHT) == [((1, 0), 1.0)]


def test_sin_ruido_el_movimiento_es_determinista():
    env = GridWorld(noise=0.0)
    assert env.transitions((3, 0), UP) == [((2, 0), 1.0)]
    assert env.transitions((3, 0), RIGHT) == [((3, 1), 1.0)]


def test_con_ruido_aparecen_las_perpendiculares():
    env = GridWorld(noise=0.2)
    outcomes = dict(env.transitions((3, 1), UP))
    assert outcomes[(2, 1)] == pytest.approx(0.8)
    assert outcomes[(3, 0)] == pytest.approx(0.1)
    assert outcomes[(3, 2)] == pytest.approx(0.1)


def test_la_misma_semilla_produce_la_misma_trayectoria():
    env = GridWorld()
    trayectorias = []
    for _ in range(2):
        obs, _ = env.reset(seed=1234)
        pasos = [obs]
        for accion in (UP, UP, RIGHT, RIGHT, DOWN, LEFT):
            obs, _, terminated, truncated, _ = env.step(accion)
            pasos.append(obs)
            if terminated or truncated:
                break
        trayectorias.append(pasos)
    assert trayectorias[0] == trayectorias[1]


def test_el_episodio_termina_al_entrar_en_un_terminal():
    env = GridWorld(noise=0.0, start=(0, 3))
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(RIGHT)
    assert terminated is True
    assert reward == pytest.approx(1.0)


def test_el_episodio_se_trunca_si_se_alarga_demasiado():
    env = GridWorld(noise=0.0, max_steps=5)
    env.reset(seed=0)
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(LEFT)  # choca contra el borde
        if terminated or truncated:
            break
    assert truncated is True


def test_el_ruido_invalido_se_rechaza():
    with pytest.raises(ValueError):
        GridWorld(noise=1.5)


def test_render_devuelve_texto():
    env = GridWorld()
    texto = env.render_values(np.zeros(env.n_states))
    assert "###" in texto
    assert len(texto.splitlines()) == env.n_rows
