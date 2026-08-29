"""Pruebas de los metodos sin modelo."""

from __future__ import annotations

import numpy as np
import pytest

from rlrs.dp import value_iteration
from rlrs.envs import GridWorld, acantilado
from rlrs.evaluation import evaluate
from rlrs.policies import EpsilonAvidaPolicy, GreedyTabularPolicy, RandomPolicy
from rlrs.td import error_frente_a, mc_control, q_learning, sarsa

METODOS = (mc_control, sarsa, q_learning)


@pytest.mark.parametrize("metodo", METODOS)
def test_forma_de_la_salida(metodo):
    env = GridWorld()
    res = metodo(env, episodes=200, seed=0)
    assert res.q.shape == (env.n_states, env.n_actions)
    assert res.retornos.shape == (200,)
    assert res.politica.shape == (env.n_states,)
    assert res.episodios == 200


@pytest.mark.parametrize("metodo", METODOS)
def test_misma_semilla_mismo_resultado(metodo):
    """Sin esto no se puede comparar nada entre corridas ni entre personas."""
    a = metodo(GridWorld(), episodes=300, seed=7)
    b = metodo(GridWorld(), episodes=300, seed=7)
    assert np.allclose(a.q, b.q)
    assert np.allclose(a.retornos, b.retornos)


@pytest.mark.parametrize("metodo", METODOS)
def test_semillas_distintas_dan_resultados_distintos(metodo):
    a = metodo(GridWorld(), episodes=300, seed=1)
    b = metodo(GridWorld(), episodes=300, seed=2)
    assert not np.allclose(a.q, b.q)


@pytest.mark.parametrize("metodo", METODOS)
def test_aprende_algo_util(metodo):
    """La politica aprendida tiene que batir con holgura a la aleatoria."""
    res = metodo(GridWorld(), episodes=3000, seed=0)
    aprendida = evaluate(GridWorld(), GreedyTabularPolicy(res.politica), episodes=200, base_seed=0)
    azar = evaluate(GridWorld(), RandomPolicy(4, seed=0), episodes=200, base_seed=0)
    assert aprendida.mean > 0.4
    assert aprendida.mean > azar.mean + 2.0
    assert aprendida.success_rate > 0.9


@pytest.mark.parametrize("metodo", METODOS)
def test_se_acerca_a_la_solucion_optima(metodo):
    """Con modelo sabemos la respuesta; sin modelo hay que quedarse cerca."""
    env = GridWorld()
    valores, _, _ = value_iteration(GridWorld(), gamma=0.9)
    res = metodo(GridWorld(), episodes=5000, seed=0)
    assert error_frente_a(res.q, valores, env) < 0.8


def test_q_learning_estima_mejor_que_monte_carlo():
    """Diferencias temporales sesga, pero tiene mucha menos varianza."""
    env = GridWorld()
    valores, _, _ = value_iteration(GridWorld(), gamma=0.9)
    mc = error_frente_a(mc_control(GridWorld(), episodes=5000, seed=0).q, valores, env)
    ql = error_frente_a(q_learning(GridWorld(), episodes=5000, seed=0).q, valores, env)
    assert ql < mc


def test_no_se_mira_el_modelo_del_entorno():
    """La frontera del Bloque 1: sin modelo significa sin `transitions()`.

    Se comprueba sobre el arbol sintactico, no espiando llamadas, porque
    `env.step()` SI usa el modelo por dentro: el entorno conoce su propia
    dinamica y la necesita para sortear. Lo que no puede conocerla es el
    agente. Espiar el metodo confundiria las dos cosas, y mirar el texto del
    archivo daria un falso positivo con la primera vez que la documentacion
    nombra el metodo.
    """
    import ast
    import inspect

    import rlrs.td

    arbol = ast.parse(inspect.getsource(rlrs.td))
    accesos = [
        n for n in ast.walk(arbol)
        if isinstance(n, ast.Attribute) and n.attr == "transitions"
    ]
    assert not accesos, "rlrs.td no puede consultar el modelo del entorno"


def test_epsilon_decae_de_punta_a_punta():
    """El decaimiento va del valor inicial al final, sin pasarse por el camino."""
    from rlrs.td import _epsilon_de

    assert _epsilon_de(0, 1000, 0.3, 0.05) == pytest.approx(0.30)
    assert _epsilon_de(999, 1000, 0.3, 0.05) == pytest.approx(0.05)
    medio = _epsilon_de(500, 1000, 0.3, 0.05)
    assert 0.05 < medio < 0.30
    # Con un solo episodio no hay recorrido: se usa el valor final.
    assert _epsilon_de(0, 1, 0.3, 0.05) == pytest.approx(0.05)


def test_epsilon_cero_solo_toma_acciones_avidas():
    """Sin exploracion, toda accion elegida tiene que ser un maximo de Q."""
    from rlrs.td import _epsilon_avida

    rng = np.random.default_rng(0)
    q = np.array([1.0, 5.0, 5.0, 2.0])
    elegidas = {_epsilon_avida(q, 0.0, rng) for _ in range(200)}
    assert elegidas <= {1, 2}, "con epsilon = 0 no puede salir una accion peor"
    # Y el desempate reparte entre las dos mejores, no se queda con la primera.
    assert elegidas == {1, 2}


def test_acantilado_bien_formado():
    env = acantilado()
    assert env.n_rows == 4 and env.n_cols == 12
    assert env.start == (3, 0)
    assert env.noise == 0.0
    assert env.step_reward == -1.0
    # Diez casillas de precipicio y una meta, todas en la fila de abajo.
    precipicio = [p for p, r in env.terminals.items() if r < 0]
    assert len(precipicio) == 10
    assert all(p[0] == 3 for p in precipicio)
    assert env.terminals[(3, 11)] == 0.0


def test_en_el_acantilado_sarsa_es_mas_prudente():
    """El resultado clasico: SARSA se aleja del borde, Q-learning se pega.

    Se comprueba por donde pasan, no por el retorno, que es lo que de verdad
    distingue a los dos metodos.
    """
    env = acantilado()
    s = sarsa(acantilado(), episodes=8000, gamma=0.99, alpha=0.5, epsilon=0.1, epsilon_final=None, seed=0)
    q = q_learning(acantilado(), episodes=8000, gamma=0.99, alpha=0.5, epsilon=0.1, epsilon_final=None, seed=0)
    # En la casilla de arranque justo encima del precipicio, (2,1):
    # Q-learning avanza pegado al borde, SARSA sube.
    i = env.state_index((2, 1))
    assert q.politica[i] == 1, "Q-learning deberia ir a la derecha, pegado al precipicio"
    assert s.politica[i] != 1, "SARSA no deberia pegarse al precipicio"


# ── la politica con la que se entrena ────────────────────────────────────────


class TestEpsilonAvidaPolicy:
    """La distincion entre entrenar y medir es el eje de la sesion 2, asi que
    la politica de entrenamiento tiene pruebas propias."""

    def test_con_epsilon_cero_es_avida(self):
        q = np.array([[0.0, 1.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        pol = EpsilonAvidaPolicy(q, epsilon=0.0, seed=0)
        assert [pol.act(0) for _ in range(20)] == [1] * 20
        assert [pol.act(1) for _ in range(20)] == [0] * 20

    def test_con_epsilon_uno_no_mira_la_tabla(self):
        q = np.array([[0.0, 1.0, 0.0, 0.0]])
        pol = EpsilonAvidaPolicy(q, epsilon=1.0, seed=0)
        elegidas = {pol.act(0) for _ in range(200)}
        assert elegidas == {0, 1, 2, 3}

    def test_desempata_al_azar(self):
        """Con argmax a secas, una fila de ceros elegiria siempre la accion 0, y
        eso sesga cualquier medicion sobre estados sin visitar."""
        pol = EpsilonAvidaPolicy(np.zeros((1, 4)), epsilon=0.0, seed=0)
        assert {pol.act(0) for _ in range(200)} == {0, 1, 2, 3}

    def test_es_reproducible(self):
        q = np.array([[0.0, 1.0, 0.0, 0.0]])
        a = [EpsilonAvidaPolicy(q, 0.5, seed=7).act(0) for _ in range(30)]
        b = [EpsilonAvidaPolicy(q, 0.5, seed=7).act(0) for _ in range(30)]
        assert a == b

    def test_reset_vuelve_a_la_misma_secuencia(self):
        pol = EpsilonAvidaPolicy(np.array([[0.0, 1.0, 0.0, 0.0]]), 0.5, seed=3)
        a = [pol.act(0) for _ in range(30)]
        pol.reset()
        assert [pol.act(0) for _ in range(30)] == a

    def test_rechaza_epsilon_fuera_de_rango(self):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonAvidaPolicy(np.zeros((2, 4)), epsilon=1.5)

    def test_rechaza_una_tabla_que_no_es_tabla(self):
        with pytest.raises(ValueError, match="estados por acciones"):
            EpsilonAvidaPolicy(np.zeros(4), epsilon=0.1)

    def test_medir_explorando_da_peor_que_medir_avido(self):
        """El error plantado de la sesion, convertido en prueba: la misma tabla
        Q medida con exploracion siempre sale peor."""
        env = GridWorld()
        ap = sarsa(env, episodes=800, gamma=0.9, seed=0)
        avido = evaluate(GridWorld(), GreedyTabularPolicy(ap.q.argmax(axis=1)),
                         episodes=200, base_seed=0)
        explorando = evaluate(GridWorld(), EpsilonAvidaPolicy(ap.q, 0.3),
                              episodes=200, base_seed=0)
        assert explorando.mean < avido.mean
