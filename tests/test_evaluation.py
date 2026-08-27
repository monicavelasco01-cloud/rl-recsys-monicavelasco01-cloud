"""Pruebas del arnes de evaluacion."""

import pytest

from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
from rlrs.evaluation import compare, evaluate
from rlrs.policies import GreedyTabularPolicy, Policy, RandomPolicy


def test_la_politica_aleatoria_cumple_el_contrato():
    politica = RandomPolicy(n_actions=4, seed=0)
    assert isinstance(politica, Policy)


def test_la_evaluacion_es_reproducible():
    env = GridWorld()
    a = evaluate(env, RandomPolicy(4, seed=0), episodes=30, base_seed=7)
    b = evaluate(env, RandomPolicy(4, seed=0), episodes=30, base_seed=7)
    assert a.mean == pytest.approx(b.mean)


def test_la_politica_optima_le_gana_a_la_aleatoria():
    """Si esto falla, algo esta roto: es la comprobacion de cordura del arnes."""
    env = GridWorld()
    _, tabla, _ = value_iteration(env, gamma=0.9)
    optima = evaluate(env, GreedyTabularPolicy(tabla, name="optima"), episodes=200)
    azar = evaluate(env, RandomPolicy(4, seed=0), episodes=200)
    assert optima.mean > azar.mean
    assert optima.success_rate > azar.success_rate


def test_el_intervalo_contiene_la_media():
    env = GridWorld()
    res = evaluate(env, RandomPolicy(4, seed=1), episodes=50)
    bajo, alto = res.ci95
    assert bajo <= res.mean <= alto


def test_compare_ordena_de_mejor_a_peor():
    env = GridWorld()
    _, tabla, _ = value_iteration(env, gamma=0.9)
    resultados = compare(
        env,
        [RandomPolicy(4, seed=0), GreedyTabularPolicy(tabla, name="optima")],
        episodes=80,
    )
    assert resultados[0].policy_name == "optima"


def test_episodios_invalidos_se_rechazan():
    env = GridWorld()
    with pytest.raises(ValueError):
        evaluate(env, RandomPolicy(4), episodes=0)
