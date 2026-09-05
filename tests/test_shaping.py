"""Pruebas del moldeado de recompensa.

La prueba que de verdad importa es la ultima: comprueba que el moldeado por
potencial suma cero a lo largo de un ciclo cerrado. Esa es la razon por la que
no cambia la politica optima, y es comprobable sin entrenar nada.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrs.envs import GridWorld
from rlrs.shaping import (
    GridWorldMoldeado,
    cercania_a_meta,
    distancia_a_meta,
    retorno_verdadero,
)


def entorno() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


def test_cercania_vale_uno_en_la_meta():
    env = entorno()
    phi = cercania_a_meta(env)
    meta = next(pos for pos, r in env.terminals.items() if r > 0)
    assert phi(meta) == pytest.approx(1.0)


def test_cercania_baja_con_la_distancia():
    env = entorno()
    phi = cercania_a_meta(env)
    meta = next(pos for pos, r in env.terminals.items() if r > 0)
    lejos = (meta[0] + 2, meta[1])
    cerca = (meta[0] + 1, meta[1])
    assert phi(cerca) > phi(lejos)


def test_modo_desconocido_falla():
    with pytest.raises(ValueError, match="modo desconocido"):
        GridWorldMoldeado(entorno(), modo="creativo")  # type: ignore[arg-type]


def test_el_entorno_original_no_se_toca():
    base = entorno()
    env = GridWorldMoldeado(base, modo="ingenuo")
    env.reset(seed=0)
    env.step(0)
    s, _ = base.reset(seed=0)
    _, r, _, _, _ = base.step(0)
    assert r == pytest.approx(-0.04) or base.is_terminal(base.state_pos(s))


def test_info_conserva_la_recompensa_original():
    env = GridWorldMoldeado(entorno(), modo="ingenuo", escala=0.5)
    env.reset(seed=3)
    _, r, _, _, info = env.step(0)
    assert r == pytest.approx(info["recompensa_original"] + info["moldeado"])


def test_ingenuo_paga_por_estar_quieto_cerca_de_la_meta():
    """El premio ingenuo no depende de haberse movido. Ahi esta el agujero."""
    env = GridWorldMoldeado(entorno(), modo="ingenuo", escala=0.5)
    meta = next(pos for pos, r in env.terminals.items() if r > 0)
    vecina = (meta[0] + 1, meta[1])
    quieto = env.extra(vecina, vecina)
    assert quieto > 0.0


def test_potencial_no_paga_por_estar_quieto():
    """Quedarse donde uno esta no vale nada, porque el premio es una diferencia."""
    env = GridWorldMoldeado(entorno(), modo="potencial", gamma=0.9, escala=0.5)
    meta = next(pos for pos, r in env.terminals.items() if r > 0)
    vecina = (meta[0] + 1, meta[1])
    quieto = env.extra(vecina, vecina)
    assert quieto < 0.0


@pytest.mark.parametrize("phi_fn", [cercania_a_meta, distancia_a_meta])
def test_potencial_suma_cero_en_un_ciclo_cerrado(phi_fn):
    """Con gamma = 1 y un recorrido que vuelve al punto de partida, el extra
    total es exactamente cero. Es la propiedad telescopica, y es la unica razon
    por la que el moldeado por potencial es seguro."""
    base = entorno()
    env = GridWorldMoldeado(base, modo="potencial", phi=phi_fn(base), gamma=1.0, escala=0.5)
    ciclo = [(3, 0), (2, 0), (2, 1), (3, 1), (3, 0)]
    total = sum(env.extra(a, b) for a, b in zip(ciclo, ciclo[1:]))
    assert total == pytest.approx(0.0, abs=1e-12)


def test_ingenuo_paga_por_dar_vueltas():
    """El mismo ciclo, con el moldeado ingenuo, deja dinero sobre la mesa."""
    base = entorno()
    env = GridWorldMoldeado(base, modo="ingenuo", gamma=1.0, escala=0.5)
    ciclo = [(3, 0), (2, 0), (2, 1), (3, 1), (3, 0)]
    total = sum(env.extra(a, b) for a, b in zip(ciclo, ciclo[1:]))
    assert total > 0.0


def test_retorno_verdadero_reproduce_con_la_misma_semilla():
    env_a, env_b = entorno(), entorno()
    pol = np.zeros(env_a.n_states, dtype=int)
    assert retorno_verdadero(env_a, pol, episodios=20, base_seed=7) == retorno_verdadero(
        env_b, pol, episodios=20, base_seed=7
    )
