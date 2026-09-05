"""Arnes del reto 4: comprueba que tu agente no se rinde.

    uv run python scripts/reto4.py

Entrena tu agente con varias semillas y mira dos cosas:

1. Que **ninguna** semilla colapse. Una sola que colapse invalida el arreglo,
   porque en tu proyecto no vas a poder elegir la semilla buena.
2. Que el retorno medio sea razonable. No hace falta llegar al optimo: hace
   falta que aprenda algo y no se rinda.

El listón sale de medir. El actor-critico tal cual colapsa en 3 de 6 semillas
y saca -1,7134 de media. REINFORCE a secas no colapsa nunca y saca +0,5413.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from retos.reto4 import mi_agente  # noqa: E402

from rlrs.aprox import caracteristicas_posicion  # noqa: E402
from rlrs.dp import value_iteration  # noqa: E402
from rlrs.envs import GridWorld  # noqa: E402
from rlrs.pg import softmax  # noqa: E402
from rlrs.shaping import retorno_verdadero  # noqa: E402

EPISODIOS = 1200
GAMMA = 0.9
SEMILLAS = range(6)
UMBRAL_COLAPSO = -1.0
MINIMO_MEDIA = 0.45


def rejilla() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


def main() -> int:
    print("\n  Reto 4 · Un agente que se rinde\n")
    _, politica_optima, _ = value_iteration(rejilla(), gamma=GAMMA)
    opt, _, _ = retorno_verdadero(rejilla(), politica_optima)

    reales, concentraciones = [], []
    for semilla in SEMILLAS:
        env = rejilla()
        ap = mi_agente(env, caracteristicas_posicion(env), EPISODIOS, GAMMA, semilla)
        pol = ap.politica(rejilla(), caracteristicas_posicion(rejilla()))
        real, _, _ = retorno_verdadero(rejilla(), pol, episodios=100)
        X = np.stack([caracteristicas_posicion(rejilla())(s) for s in range(env.n_states)])
        p = softmax(ap.red.adelante(X))
        reales.append(real)
        concentraciones.append(float(p.max(axis=1).mean()))
        marca = "COLAPSA" if real < UMBRAL_COLAPSO else ""
        print(f"     semilla {semilla}   retorno {real:>+8.4f}   "
              f"concentracion {concentraciones[-1]:.3f}   {marca}")

    a = np.array(reales)
    colapsos = int((a < UMBRAL_COLAPSO).sum())
    print(f"\n  Media                 {a.mean():>+8.4f}   hace falta >= {MINIMO_MEDIA:+.2f}")
    print(f"  Peor semilla          {a.min():>+8.4f}")
    print(f"  Colapsos              {colapsos:>8d}   hace falta 0")
    print(f"  Referencia: la politica optima saca {opt:+.4f}\n")

    if colapsos == 0 and a.mean() >= MINIMO_MEDIA:
        print("  SUPERADO.  Tu agente aprende en las seis semillas y ninguna se rinde.")
        print("             Comprueba que tu bitacora explica POR QUE funciona: el")
        print("             arreglo sin el diagnostico no cuenta.\n")
        return 0

    print("  NO SUPERADO\n")
    if colapsos:
        print(f"  {colapsos} de {len(a)} semillas se rindieron. Mira la concentracion de esas:")
        print("  si esta muy por encima de 0,25, que es lo que daria el azar, la")
        print("  politica se cerro antes de encontrar nada.")
    if a.mean() < MINIMO_MEDIA:
        print(f"  La media ({a.mean():+.4f}) no llega al minimo. Puede que hayas frenado")
        print("  el colapso a costa de que no aprenda: mantener la politica al azar")
        print("  evita rendirse y tampoco resuelve el problema.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
