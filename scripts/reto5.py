"""Arnes del reto 5: comprueba que tu politica no se queda clavada.

    uv run python scripts/reto5.py

Mide tu politica sobre Meridiano visto como bandido, a cuatro horizontes, y
comprueba dos cosas:

1. Que **gane de verdad**, no solo que gane mas que la avida.
2. Que **reparta las tiradas**. Una politica que toca tres brazos de veinte no
   esta resolviendo el problema aunque el numero salga bien.

El liston sale de medir, no de opinar, y hay una sorpresa dentro: se imprime al
final.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from retos.reto5 import mi_politica  # noqa: E402

from rlrs.bandidos import Corrida, MeridianoBandido, epsilon_avida  # noqa: E402
from rlrs.meridiano import ajustar_pfa  # noqa: E402

CATALOGO = 20
HORIZONTES = (25, 100, 500, 1500)
RONDAS = 500
SEMILLAS = 10
MINIMO_GANANCIA = 0.1400
MINIMO_BRAZOS = 15.0


def ic(xs) -> tuple[float, float, float]:
    a = np.asarray(xs, dtype=float)
    h = 1.96 * float(a.std(ddof=1)) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(a.mean()) - h, float(a.mean()) + h


def _azar(b, n, s) -> Corrida:
    rng = np.random.default_rng(s)
    brazos, recompensas = [], []
    for _ in range(n):
        a = int(rng.integers(b.n_brazos))
        recompensas.append(b.tirar(a, rng))
        brazos.append(a)
    return Corrida("al azar", np.array(brazos), np.array(recompensas), float("nan"))


def medir(modelo, hacer, rondas: int, semillas: int = SEMILLAS):
    ganancias, brazos = [], []
    for s in range(semillas):
        b = MeridianoBandido(modelo, catalogo=CATALOGO, rondas=rondas, recompensa="dominio")
        b.reset(seed=s)
        antes = b.env.dominio()
        corrida = hacer(b, rondas, s)
        if corrida.rondas != rondas:
            raise SystemExit(
                f"\n  Tu politica jugo {corrida.rondas} rondas y te habian dado {rondas}.\n"
                "  El arnes compara con el mismo presupuesto para todas.\n")
        ganancias.append(b.env.dominio() - antes)
        brazos.append(int((corrida.reparto(CATALOGO) > 0).sum()))
    return ganancias, brazos


def main() -> int:
    print("\n  Reto 5 · La politica que se queda clavada\n")
    print("  Ajustando el modelo de estudiante...", flush=True)
    modelo = ajustar_pfa()

    print("\n  Tu politica, a cuatro horizontes\n")
    print(f"  {'rondas':>7s}  {'ganancia':>10s}  {'IC 95 %':>22s}  {'brazos':>10s}")
    for n in HORIZONTES:
        g, b = medir(modelo, mi_politica, n)
        m, lo, hi = ic(g)
        print(f"  {n:>7d}  {m:>+10.4f}  [{lo:+8.4f}, {hi:+8.4f}]  {np.mean(b):>7.1f}/20")

    g, brazos = medir(modelo, mi_politica, RONDAS)
    media, lo, hi = ic(g)
    media_brazos = float(np.mean(brazos))

    print(f"\n  Criterios, a {RONDAS} rondas\n")
    ok_g = media >= MINIMO_GANANCIA
    ok_b = media_brazos >= MINIMO_BRAZOS
    print(f"  Ganancia media    {media:>+9.4f}   hace falta >= {MINIMO_GANANCIA:+.4f}   "
          f"{'sí' if ok_g else 'NO'}")
    print(f"  Brazos usados     {media_brazos:>9.1f}   hace falta >= {MINIMO_BRAZOS:.0f}       "
          f"{'sí' if ok_b else 'NO'}")
    print(f"  Peor semilla      {min(g):>+9.4f}")

    # Las dos referencias, medidas aqui mismo para que nadie tenga que creerselas.
    ga, _ = medir(modelo, lambda b, n, s: epsilon_avida(b, n, epsilon=0.0, seed=s), RONDAS)
    gz, _ = medir(modelo, _azar, RONDAS)
    m_avida, _, _ = ic(ga)
    m_azar, lo_azar, hi_azar = ic(gz)
    print(f"\n  Referencias, medidas ahora mismo con las mismas semillas:")
    print(f"    la avida pura, que es lo que venia de fabrica   {m_avida:+.4f}")
    print(f"    **tirar al azar**                               {m_azar:+.4f}  "
          f"[{lo_azar:+.4f}, {hi_azar:+.4f}]")

    if ok_g and ok_b:
        print("\n  SUPERADO.  Tu politica reparte las tiradas y gana de verdad.")
    else:
        print("\n  NO SUPERADO")
        if not ok_b:
            print("  Estas tocando pocos brazos. Mira que le pasa al brazo que eliges")
            print("  cuando tiras de el muchas veces seguidas.")
        if not ok_g and ok_b:
            print("  Repartes, pero no ganas lo suficiente. Puede que estes repartiendo")
            print("  demasiado: explorar tambien cuesta.")

    # La parte incomoda va al final y siempre, se haya superado o no.
    solapa = not (hi < lo_azar or hi_azar < lo)
    print("\n  ── Y ahora lo que hay que anotar en la bitacora ──")
    if media > hi_azar:
        print(f"  Tu politica ({media:+.4f}) supera al azar ({m_azar:+.4f}) y los")
        print("  intervalos no se tocan. Eso es un resultado, y es poco comun.")
    elif solapa:
        print(f"  Tu politica saca {media:+.4f} y **tirar al azar** saca {m_azar:+.4f}.")
        print("  Los intervalos se solapan: con estas semillas no puedes afirmar que")
        print("  tu politica sea mejor que elegir al azar.")
        print("  No es un fallo tuyo. Es lo que pasa cuando los brazos se agotan:")
        print("  cualquier cosa que reparta gana, y la sofisticacion aporta poco.")
        print("  Escribelo asi en el informe. Es exactamente el tipo de frase que")
        print("  distingue un trabajo honesto de uno que solo enseña lo que le sale bien.")
    else:
        print(f"  Tu politica ({media:+.4f}) queda por debajo del azar ({m_azar:+.4f}).")
        print("  Antes de tocar nada, pregunta por que una politica que no mira nada")
        print("  le gana a una que si mira.")
    print()
    return 0 if (ok_g and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
