#!/usr/bin/env python
"""Que pasa cuando se quita el descuento. Sesion 1, el error plantado.

    uv run python experiments/divergencia.py

Recorre los tres intentos de la guia, en orden:

  1. `value_iteration(env, gamma=1.0)`  ->  el codigo se niega.
  2. El mismo bucle escrito a mano, gamma = 1, entorno por defecto  ->  converge.
     Perder la garantia no es lo mismo que perder la convergencia.
  3. Igual, pero pagando +0.01 por paso en vez de cobrarlo  ->  no converge.
     Los valores crecen +0.01 por barrido, para siempre.

El bucle de aqui es a proposito una copia desnuda de `rlrs.dp.value_iteration`,
sin la guardia de gamma y sin criterio de parada. No es codigo que debas imitar:
es el laboratorio donde se rompe la teoria a mano.
"""

from __future__ import annotations

import numpy as np

from rlrs.dp import q_value, value_iteration
from rlrs.envs import GridWorld

HITOS = (1, 5, 10, 50, 100, 500, 1000, 2000)


def barrer(env: GridWorld, gamma: float, barridos: int) -> dict[int, tuple[float, float, float]]:
    """Aplica el operador de Bellman ``barridos`` veces, sin pararse nunca.

    Devuelve, para cada hito, la terna ``(V(3,0), max|V|, cambio del barrido)``.
    """
    values = np.zeros(env.n_states, dtype=float)
    registro: dict[int, tuple[float, float, float]] = {}

    for k in range(1, barridos + 1):
        nuevos = np.zeros_like(values)
        delta = 0.0
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                pos = (r, c)
                if env.is_wall(pos) or env.is_terminal(pos):
                    continue
                idx = env.state_index(pos)
                mejor = max(q_value(env, values, pos, a, gamma) for a in range(env.n_actions))
                delta = max(delta, abs(mejor - values[idx]))
                nuevos[idx] = mejor
        values = nuevos
        if k in HITOS:
            registro[k] = (float(values[env.state_index((3, 0))]), float(np.abs(values).max()), delta)
    return registro


def tabla(titulo: str, registro: dict[int, tuple[float, float, float]]) -> None:
    print(f"\n  {titulo}")
    print(f"  {'barrido':>8} {'V(3,0)':>12} {'max|V|':>12} {'cambio':>12}")
    print("  " + "-" * 48)
    for k, (v, mx, d) in registro.items():
        print(f"  {k:>8} {v:>12.4f} {mx:>12.4f} {d:>12.6f}")


def main() -> None:
    print("\n  UPTC · Sesion 1 · Que sostiene la convergencia\n")

    # ── Intento 1 ───────────────────────────────────────────────────────────
    print("  1) value_iteration(env, gamma=1.0)")
    try:
        value_iteration(GridWorld(), gamma=1.0)
        print("     No deberias estar leyendo esto: la guardia no salto.")
    except ValueError as exc:
        print(f"     ValueError: {exc}")
    print("     La guardia protege una garantia: sin gamma < 1 el operador de")
    print("     Bellman deja de ser una contraccion.")

    # ── Intento 2 ───────────────────────────────────────────────────────────
    tabla(
        "2) gamma = 1.0, recompensa por paso -0.04  (el entorno de siempre)",
        barrer(GridWorld(), gamma=1.0, barridos=2000),
    )
    print("     Converge. Quedarse dando vueltas cuesta -0.04 por paso, o sea")
    print("     -infinito, asi que ninguna politica que el max prefiera lo hace.")
    print("     Es un camino mas corto estocastico, y ahi gamma = 1 esta bien")
    print("     definido. Perder la garantia no es perder la convergencia.")

    # ── Intento 3 ───────────────────────────────────────────────────────────
    env_pagado = GridWorld(step_reward=+0.01)
    tabla(
        "3) gamma = 1.0, recompensa por paso +0.01  (ahora le pagamos por moverse)",
        barrer(env_pagado, gamma=1.0, barridos=2000),
    )
    print("     No converge. A partir del barrido 100 los valores crecen +0.01")
    print("     por barrido, indefinidamente, y el cambio se estanca en 0.01:")
    print("     el criterio de parada nunca se dispara.")

    # ── El contraste ────────────────────────────────────────────────────────
    valores, _, barridos = value_iteration(env_pagado, gamma=0.9)
    print(f"\n  El mismo entorno con gamma = 0.9 converge en {barridos} barridos,")
    print(f"  con max|V| = {np.abs(valores).max():.4f}. La unica diferencia es el descuento.")

    print("\n  El diagnostico tiene dos capas.")
    print("    Matematica: con gamma = 1 existe una politica que nunca termina y")
    print("    acumula +0.01 sin fin, luego su retorno es +infinito. No hay punto")
    print("    fijo finito al que converger.")
    print("    De diseno: el error no fue poner gamma = 1, fue pagar por el")
    print("    proceso en vez de por el resultado. El descuento solo lo tapaba.\n")


if __name__ == "__main__":
    main()
