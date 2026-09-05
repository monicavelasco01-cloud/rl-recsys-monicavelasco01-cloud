"""Pruebas del contrato del reto 4.

Comprueban la forma de la respuesta, no su calidad: si el agente aprende bien
lo decide ``scripts/reto4.py``, que lo mide con varias semillas.
"""

from __future__ import annotations

import numpy as np
import pytest

from retos.reto4 import mi_agente

from rlrs.aprox import caracteristicas_posicion
from rlrs.envs import GridWorld
from rlrs.pg import EntrenamientoPG, actor_critico, softmax


def entorno() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


def _por_defecto() -> bool:
    """True mientras la respuesta sea la que venia de fabrica."""
    env = entorno()
    phi = caracteristicas_posicion(env)
    mio = mi_agente(entorno(), phi, 30, 0.9, 0)
    base = actor_critico(entorno(), phi, episodes=30, gamma=0.9, seed=0)
    return bool(np.allclose(mio.red.parametros()[0], base.red.parametros()[0]))


pytestmark = pytest.mark.skipif(
    _por_defecto(),
    reason="El reto 4 todavia no esta contestado: modifica 'mi_agente' en retos/reto4.py",
)


def test_devuelve_un_entrenamiento():
    env = entorno()
    r = mi_agente(env, caracteristicas_posicion(env), 40, 0.9, 0)
    assert isinstance(r, EntrenamientoPG)


def test_respeta_el_numero_de_episodios():
    """Entrenar mas episodios de los que te dan no es arreglar el problema."""
    env = entorno()
    r = mi_agente(env, caracteristicas_posicion(env), 40, 0.9, 0)
    assert len(r.retornos) == 40, (
        f"Devolviste {len(r.retornos)} episodios y te habian dado 40. El arnes "
        "compara con el mismo presupuesto para todos."
    )


def test_respeta_la_semilla():
    env = entorno()
    a = mi_agente(entorno(), caracteristicas_posicion(env), 40, 0.9, 7)
    b = mi_agente(entorno(), caracteristicas_posicion(env), 40, 0.9, 7)
    assert np.allclose(a.red.parametros()[0], b.red.parametros()[0]), (
        "Dos llamadas con la misma semilla dan resultados distintos. Sin eso, "
        "nada de lo que midas es reproducible."
    )


def test_la_politica_no_se_cierra_del_todo():
    """La concentracion es el sintoma que se veia antes del colapso."""
    env = entorno()
    r = mi_agente(env, caracteristicas_posicion(env), 600, 0.9, 0)
    X = np.stack([caracteristicas_posicion(entorno())(s) for s in range(env.n_states)])
    concentracion = float(softmax(r.red.adelante(X)).max(axis=1).mean())
    assert concentracion < 0.97, (
        f"La accion mas probable se lleva el {concentracion:.1%} de media. Una "
        "politica asi de cerrada ya no explora, y si deja de llegar a la meta no "
        "tiene forma de enterarse."
    )
