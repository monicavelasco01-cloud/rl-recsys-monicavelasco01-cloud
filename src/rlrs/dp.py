"""Programacion dinamica: resolver un MDP cuando *si* conocemos el modelo.

Esto es el laboratorio 4 del modulo, adelantado aqui en su version minima para
que la plantilla tenga algo real que probar. La ecuacion que implementa
:func:`value_iteration` es la de optimalidad de Bellman:

    V(s) <- max_a  sum_s'  P(s'|s,a) [ R(s') + gamma * V(s') ]

Leida en voz alta: *el valor de un estado es, para la mejor accion disponible,
la recompensa que espero recibir mas el valor descontado de donde acabare*.
"""

from __future__ import annotations

import numpy as np

from rlrs.envs import GridWorld


def q_value(env: GridWorld, values: np.ndarray, pos: tuple[int, int], action: int, gamma: float) -> float:
    """Valor de tomar ``action`` en ``pos`` y despues seguir la politica avida."""
    total = 0.0
    for nxt, prob in env.transitions(pos, action):
        v_next = 0.0 if env.is_terminal(nxt) else values[env.state_index(nxt)]
        total += prob * (env.reward_of(nxt) + gamma * v_next)
    return total


def value_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    tol: float = 1e-8,
    max_sweeps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Itera la ecuacion de optimalidad hasta que el cambio maximo baja de ``tol``.

    Returns
    -------
    values:
        Vector ``V`` con un valor por estado (0 en muros y terminales).
    policy:
        Vector con la accion avida de cada estado.
    sweeps:
        Barridos necesarios para converger. Util para ver como depende de
        ``gamma``: cuanto mas cerca de 1, mas barridos hacen falta.

    Notes
    -----
    La convergencia esta garantizada porque el operador de Bellman es una
    contraccion de modulo ``gamma`` en la norma del supremo. Es decir: cada
    barrido acerca la estimacion al punto fijo al menos en un factor ``gamma``.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma debe estar en [0, 1); se recibio {gamma}")

    values = np.zeros(env.n_states, dtype=float)
    sweeps = 0

    for sweeps in range(1, max_sweeps + 1):
        new_values = np.zeros_like(values)
        delta = 0.0
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                pos = (r, c)
                if env.is_wall(pos) or env.is_terminal(pos):
                    continue
                idx = env.state_index(pos)
                best = max(q_value(env, values, pos, a, gamma) for a in range(env.n_actions))
                delta = max(delta, abs(best - values[idx]))
                new_values[idx] = best
        values = new_values
        if delta < tol:
            break

    policy = greedy_policy(env, values, gamma)
    return values, policy, sweeps


def greedy_policy(env: GridWorld, values: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """Accion avida en cada estado, dada una funcion de valor."""
    policy = np.zeros(env.n_states, dtype=int)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos = (r, c)
            if env.is_wall(pos) or env.is_terminal(pos):
                continue
            q_values = [q_value(env, values, pos, a, gamma) for a in range(env.n_actions)]
            policy[env.state_index(pos)] = int(np.argmax(q_values))
    return policy
