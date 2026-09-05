"""Pruebas del entorno del proyecto.

La mas importante es la ultima: comprueba que la recompensa por aciertos
premia a una politica que ensena poco. Si algun dia deja de fallar, es que el
simulador dejo de reproducir el problema que el proyecto quiere plantear.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrs.meridiano import Meridiano, ModeloPFA, medir


def modelo_de_juguete(k: int = 5) -> ModeloPFA:
    """Un modelo inventado, para que las pruebas no dependan de los datos.

    La habilidad 0 es facil (beta alto) y no ensena nada (gamma cero); la 4 es
    dificil y ensena mucho. Con eso basta para probar el mecanismo.
    """
    beta = np.linspace(2.0, -2.0, k)
    gamma = np.linspace(0.0, 1.0, k)
    rho = np.zeros(k)
    return ModeloPFA(beta, gamma, rho, {i: f"habilidad {i}" for i in range(k)})


def entorno(recompensa: str = "dominio") -> Meridiano:
    return Meridiano(modelo_de_juguete(), recompensa=recompensa, n_pasos=20, catalogo=5)


def test_recompensa_desconocida_falla():
    with pytest.raises(ValueError, match="recompensa desconocida"):
        Meridiano(modelo_de_juguete(), recompensa="puntos")  # type: ignore[arg-type]


def test_el_estado_son_aciertos_y_fallos():
    env = entorno()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2 * env.n_actions,)
    assert not obs.any()


def test_cada_paso_incrementa_aciertos_o_fallos():
    env = entorno()
    env.reset(seed=0)
    obs, _, _, _, info = env.step(0)
    assert obs.sum() == 1.0
    assert obs[0] == float(info["acierto"])


def test_el_episodio_dura_lo_que_dice():
    env = entorno()
    env.reset(seed=0)
    for i in range(env.n_pasos - 1):
        _, _, fin, _, _ = env.step(0)
        assert not fin
    _, _, fin, _, _ = env.step(0)
    assert fin


def test_reproduce_con_la_misma_semilla():
    a, b = entorno(), entorno()
    a.reset(seed=7)
    b.reset(seed=7)
    ra = [a.step(i % a.n_actions)[1] for i in range(20)]
    rb = [b.step(i % b.n_actions)[1] for i in range(20)]
    assert np.allclose(ra, rb)


def test_la_recompensa_por_aciertos_es_cero_o_uno():
    env = entorno("aciertos")
    env.reset(seed=0)
    for i in range(10):
        _, r, _, _, _ = env.step(i % env.n_actions)
        assert r in (0.0, 1.0)


def test_el_dominio_no_baja_al_practicar():
    """Con rho = 0, fallar no ensena pero tampoco desaprende."""
    env = entorno()
    _, info = env.reset(seed=1)
    antes = info["dominio"]
    for i in range(20):
        _, _, _, _, info = env.step(i % env.n_actions)
    assert info["dominio"] >= antes - 1e-9


def test_la_politica_facil_acierta_mucho_y_ensena_poco():
    """El nucleo del proyecto, comprobado sin depender de los datos reales.

    La habilidad 0 es la mas facil y no ensena nada. Una politica que solo la
    repite consigue muchos aciertos y ninguna ganancia. Una rotacion consigue
    menos aciertos y mas ganancia. Si esto deja de cumplirse, el simulador ya
    no plantea el problema del que trata el proyecto.
    """
    env = entorno()
    turno = [-1]

    def por_turnos(obs):
        turno[0] = (turno[0] + 1) % env.n_actions
        return turno[0]

    facil = medir(env, lambda obs: 0, episodios=30)
    rota = medir(env, por_turnos, episodios=30)

    assert facil["aciertos"] > rota["aciertos"]
    assert facil["ganancia"] < rota["ganancia"]
    assert facil["habilidades_distintas"] < rota["habilidades_distintas"]
