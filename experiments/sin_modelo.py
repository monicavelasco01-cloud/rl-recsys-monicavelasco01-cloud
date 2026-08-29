#!/usr/bin/env python
"""Aprender sin modelo. Sesion 2.

    uv run python experiments/sin_modelo.py            # todo
    uv run python experiments/sin_modelo.py --parte 3  # solo una parte

Cuatro partes, en el orden de la guia:

  1. Los tres metodos contra la respuesta que ya conocemos. La programacion
     dinamica de la sesion 1 nos dio V*, asi que aqui se puede medir de verdad
     cuanto le falta a cada metodo sin modelo.
  2. Curvas de aprendizaje: quien mejora antes y quien mejora mas.
  3. El error plantado: medir con la politica que exploraba.
  4. El acantilado: donde SARSA y Q-learning por fin no coinciden.

Tarda unos dos minutos entero. Con --parte se ejecuta solo una.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlrs.dp import value_iteration
from rlrs.envs import ARROWS, GridWorld, acantilado
from rlrs.evaluation import evaluate
from rlrs.policies import EpsilonAvidaPolicy, GreedyTabularPolicy
from rlrs.td import error_frente_a, mc_control, q_learning, sarsa

METODOS = (("monte-carlo", mc_control), ("sarsa", sarsa), ("q-learning", q_learning))


# La politica con la que se entrena vive ahora en rlrs.policies, para que la
# distincion entre entrenar y medir sea visible desde la biblioteca y no solo
# dentro de este experimento.
EpsilonAvida = EpsilonAvidaPolicy


def rejilla(env: GridWorld, politica: np.ndarray) -> str:
    filas = []
    for r in range(env.n_rows):
        fila = ""
        for c in range(env.n_cols):
            pos = (r, c)
            if env.is_wall(pos):
                fila += " # "
            elif env.is_terminal(pos):
                fila += " G " if env.terminals[pos] >= 0 else " X "
            else:
                fila += " " + ARROWS[int(politica[env.state_index(pos)])] + " "
        filas.append(fila)
    return "\n".join(filas)


# ── 1 ───────────────────────────────────────────────────────────────────────

def parte1() -> None:
    print("\n  1 · Los tres metodos contra la respuesta conocida\n")
    env = GridWorld()
    valores, politica_dp, barridos = value_iteration(GridWorld(), gamma=0.9)
    print(f"  Programacion dinamica: {barridos} barridos. Esta es la referencia.")
    print(rejilla(env, politica_dp))

    print(f"\n  {'metodo':<13} {'err. max en V':>14} {'politica igual':>16} {'evaluacion avida'}")
    print("  " + "-" * 74)
    for nombre, fn in METODOS:
        res = fn(GridWorld(), episodes=5000, gamma=0.9, seed=0)
        err = error_frente_a(res.q, valores, env)
        iguales = sum(
            1
            for i in range(env.n_states)
            if not env.is_wall(env.state_pos(i))
            and not env.is_terminal(env.state_pos(i))
            and res.politica[i] == politica_dp[i]
        )
        ev = evaluate(GridWorld(), GreedyTabularPolicy(res.politica, name=nombre), episodes=300, base_seed=0)
        print(f"  {nombre:<13} {err:>14.4f} {f'{iguales}/16':>16}  {ev.mean:+.3f} [{ev.ci95[0]:+.3f}, {ev.ci95[1]:+.3f}]")

    print("\n  Monte Carlo acierta la politica entera y aun asi es el que peor")
    print("  estima los valores. La politica converge mucho antes que V.")


# ── 2 ───────────────────────────────────────────────────────────────────────

def parte2() -> None:
    print("\n  2 · Curvas de aprendizaje, por tramos de 500 episodios\n")
    curvas = {n: f(GridWorld(), episodes=5000, gamma=0.9, seed=0).retornos for n, f in METODOS}
    print(f"  {'tramo':>12} {'monte-carlo':>13} {'sarsa':>10} {'q-learning':>12}")
    print("  " + "-" * 50)
    for i in range(0, 5000, 500):
        print(
            f"  {f'{i}-{i+500}':>12} "
            f"{curvas['monte-carlo'][i:i+500].mean():>13.3f} "
            f"{curvas['sarsa'][i:i+500].mean():>10.3f} "
            f"{curvas['q-learning'][i:i+500].mean():>12.3f}"
        )
    print("\n  En los primeros 500 episodios Monte Carlo esta practicamente en cero:")
    print("  no aprende nada hasta que un episodio termina, y al principio casi")
    print("  ninguno termina dentro del limite de pasos.")


# ── 3 ───────────────────────────────────────────────────────────────────────

def parte3() -> None:
    print("\n  3 · El error plantado: medir con la politica que exploraba\n")
    res = q_learning(GridWorld(), episodes=5000, gamma=0.9, seed=0)
    ev = evaluate(GridWorld(), GreedyTabularPolicy(res.politica, name="avida"), episodes=300, base_seed=0)
    print(f"  {'como se mide':<22} {'retorno':>9}  {'IC 95%':>20} {'exito':>7}")
    print("  " + "-" * 62)
    print(f"  {'avida (epsilon = 0)':<22} {ev.mean:>+9.3f}  [{ev.ci95[0]:+.3f}, {ev.ci95[1]:+.3f}] {ev.success_rate:>7.1%}")
    for eps in (0.05, 0.1, 0.3):
        e = evaluate(GridWorld(), EpsilonAvida(res.q, eps, f"epsilon = {eps}"), episodes=300, base_seed=0)
        print(f"  {f'epsilon = {eps}':<22} {e.mean:>+9.3f}  [{e.ci95[0]:+.3f}, {e.ci95[1]:+.3f}] {e.success_rate:>7.1%}")

    print("\n  Es el MISMO agente en las cuatro filas. Lo unico que cambia es si")
    print("  sigue explorando mientras se le mide. Con epsilon = 0.3 los")
    print("  intervalos ni siquiera se solapan con los de epsilon = 0: la")
    print("  conclusion equivocada seria estadisticamente significativa.")


# ── 4 ───────────────────────────────────────────────────────────────────────

def parte4() -> None:
    print("\n  4 · El acantilado: donde SARSA y Q-learning no coinciden\n")
    print("  Nota: en esta variante la meta entrega 0, no una recompensa positiva,")
    print("  asi que el indicador de exito no aplica. Lo que se mira es el retorno.\n")
    env = acantilado()
    for nombre, fn in (("sarsa", sarsa), ("q-learning", q_learning)):
        res = fn(acantilado(), episodes=8000, gamma=0.99, alpha=0.5, epsilon=0.1, epsilon_final=None, seed=0)
        avida = evaluate(acantilado(), GreedyTabularPolicy(res.politica, name=nombre), episodes=200, base_seed=0)
        explor = evaluate(acantilado(), EpsilonAvida(res.q, 0.1, nombre), episodes=200, base_seed=0)
        print(f"  {nombre}")
        print(f"     durante el entrenamiento   {res.retornos[-1000:].mean():>+8.2f}")
        print(f"     evaluada avida             {avida.mean:>+8.2f}  [{avida.ci95[0]:+.2f}, {avida.ci95[1]:+.2f}]  {avida.lengths.mean():.0f} pasos")
        print(f"     evaluada explorando        {explor.mean:>+8.2f}  [{explor.ci95[0]:+.2f}, {explor.ci95[1]:+.2f}]  {explor.lengths.mean():.0f} pasos")
        print(rejilla(env, res.politica))
        print()

    print("  SARSA se va por arriba, lejos del borde. Q-learning pega el camino")
    print("  al precipicio. Avida, la de Q-learning es mejor. Explorando, es")
    print("  mucho peor, porque cada desvio la tira al vacio. Ninguna de las dos")
    print("  esta equivocada: optimizan cosas distintas.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos de la sesion 2")
    parser.add_argument("--parte", type=int, choices=(1, 2, 3, 4), help="ejecutar solo una parte")
    args = parser.parse_args()

    print("\n  UPTC · Sesion 2 · Aprender sin modelo del entorno")
    partes = {1: parte1, 2: parte2, 3: parte3, 4: parte4}
    for n in ([args.parte] if args.parte else (1, 2, 3, 4)):
        partes[n]()
    print()


if __name__ == "__main__":
    main()
