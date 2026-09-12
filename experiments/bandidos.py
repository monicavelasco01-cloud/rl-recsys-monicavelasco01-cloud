"""Bandidos, medidos.

    uv run python experiments/bandidos.py              las cuatro partes
    uv run python experiments/bandidos.py --parte 2    solo una

Cuatro partes:

  1. El bandido de libro. Arrepentimiento de cuatro politicas sobre un bandido
     estacionario de veinte brazos.
  2. **El horizonte cambia al ganador.** La misma comparacion a cinco
     horizontes distintos. Es el resultado central de la sesion.
  3. La constante de UCB. Que pasa cuando se usa la del teorema.
  4. Meridiano como bandido. Donde la aproximacion se rompe, y por que.

Todos los numeros de la guia del viernes salen de aqui. Si algo no coincide,
manda la salida entera al foro: puede ser un error del material.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlrs.bandidos import (
    Corrida,
    MeridianoBandido,
    bandido_dificil,
    epsilon_avida,
    thompson,
    ucb1,
)
from rlrs.meridiano import ajustar_pfa

SEMILLAS = 20
BRAZOS = 20
BRECHA = 0.05


def ic(xs) -> tuple[float, float, float]:
    a = np.asarray(xs, dtype=float)
    h = 1.96 * float(a.std(ddof=1)) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(a.mean()) - h, float(a.mean()) + h


POLITICAS = [
    ("ávida pura", lambda b, n, s: epsilon_avida(b, n, epsilon=0.0, seed=s)),
    ("ε-ávida 0,10", lambda b, n, s: epsilon_avida(b, n, epsilon=0.10, seed=s)),
    ("UCB1 c=2", lambda b, n, s: ucb1(b, n, c=2.0, seed=s)),
    ("Thompson", lambda b, n, s: thompson(b, n, seed=s)),
]


# ── 1 ───────────────────────────────────────────────────────────────────────
def parte1() -> None:
    print("\n  1 · El bandido de libro\n")
    b = bandido_dificil(BRAZOS, BRECHA, seed=0)
    orden = np.sort(b.p)[::-1]
    print(f"  {BRAZOS} brazos. El mejor tiene p = {b.p_mejor:.4f} y el segundo "
          f"{orden[1]:.4f}: la brecha es {b.p_mejor - orden[1]:.4f}.")
    print("  La dificultad de un bandido no la da el numero de brazos, la da la brecha.\n")
    print(f"  2000 rondas, {SEMILLAS} semillas\n")
    print(f"  {'':<16s} {'arrepentimiento':>16s}  {'IC 95 %':>20s}")
    for nombre, hacer in POLITICAS:
        m, lo, hi = ic([hacer(b, 2000, s).arrepentimiento(b.p) for s in range(SEMILLAS)])
        print(f"  {nombre:<16s} {m:>16.2f}  [{lo:7.2f}, {hi:7.2f}]")
    print("\n  El arrepentimiento se calcula con las probabilidades VERDADERAS.")
    print("  Fuera del laboratorio no se conocen, asi que fuera del laboratorio")
    print("  esta metrica no se puede calcular. Conviene saberlo antes de usarla.\n")


# ── 2 ───────────────────────────────────────────────────────────────────────
def parte2() -> None:
    print("\n  2 · El horizonte cambia al ganador\n")
    b = bandido_dificil(BRAZOS, BRECHA, seed=0)
    horizontes = [100, 200, 1000, 2000, 20000]
    filas = {}
    for n in horizontes:
        filas[n] = {nombre: ic([hacer(b, n, s).arrepentimiento(b.p) for s in range(SEMILLAS)])
                    for nombre, hacer in POLITICAS}

    print(f"  {'rondas':>7s}   {'ávida pura (sin explorar)':<38s} {'Thompson (explorando)':<36s}")
    for n in horizontes:
        a = filas[n]["ávida pura"]
        t = filas[n]["Thompson"]
        solapa = not (a[2] < t[1] or t[2] < a[1])
        print(f"  {n:>7d}  ávida {a[0]:9.2f} [{a[1]:8.2f},{a[2]:8.2f}]   "
              f"Thompson {t[0]:8.2f} [{t[1]:7.2f},{t[2]:7.2f}]   "
              f"{'empatan' if solapa else 'SEPARADAS'}")

    # La conclusion se calcula de la tabla, no se escribe a mano.
    primeros_separados = [n for n in horizontes
                          if not (filas[n]["ávida pura"][2] < filas[n]["Thompson"][1]
                                  or filas[n]["Thompson"][2] < filas[n]["ávida pura"][1])]
    ultimo_empate = max(primeros_separados) if primeros_separados else None
    if ultimo_empate is not None:
        siguientes = [n for n in horizontes if n > ultimo_empate]
        if siguientes:
            print(f"\n  Con {ultimo_empate} rondas las dos politicas todavia EMPATAN.")
            print(f"  Con {min(siguientes)} rondas ya no: la avida se queda atras.")
            print("  El punto de cruce esta entre esos dos numeros.\n")
    peor = filas[20000]["ávida pura"][0] / filas[20000]["Thompson"][0]
    print(f"  A 20.000 rondas la avida pura arrepiente {peor:.1f} veces mas que Thompson.\n")
    print("  Moraleja: la pregunta «cual es mejor» esta mal planteada si no dice")
    print("  cuantas rondas se van a jugar. Una politica no es mejor que otra: es")
    print("  mejor A PARTIR DE cierto horizonte.\n")


# ── 3 ───────────────────────────────────────────────────────────────────────
def parte3() -> None:
    print("\n  3 · La constante de UCB\n")
    b = bandido_dificil(BRAZOS, BRECHA, seed=0)
    n = 5000
    print(f"  {n} rondas, {SEMILLAS} semillas\n")
    print(f"  {'':<22s} {'arrepentimiento':>16s}  {'IC 95 %':>20s}")
    for c in (0.25, 0.5, 1.0, 2.0):
        m, lo, hi = ic([ucb1(b, n, c=c, seed=s).arrepentimiento(b.p) for s in range(SEMILLAS)])
        marca = "  <- la del teorema" if c == 2.0 else ""
        print(f"  UCB1 c={c:<16g} {m:>16.2f}  [{lo:7.2f}, {hi:7.2f}]{marca}")
    m, lo, hi = ic([epsilon_avida(b, n, epsilon=0.10, seed=s).arrepentimiento(b.p)
                    for s in range(SEMILLAS)])
    print(f"  {'ε-ávida 0,10':<22s} {m:>16.2f}  [{lo:7.2f}, {hi:7.2f}]")
    print("\n  UCB con la constante de su propio teorema pierde contra una epsilon-avida")
    print("  con epsilon fijo. No es un error: la cota es asintotica y vale para el")
    print("  peor caso. Su problema no es el peor caso y usted no vive en el infinito.\n")


# ── 4 ───────────────────────────────────────────────────────────────────────
def _azar(b, n, s) -> Corrida:
    rng = np.random.default_rng(s)
    brazos, recompensas = [], []
    for _ in range(n):
        a = int(rng.integers(b.n_brazos))
        recompensas.append(b.tirar(a, rng))
        brazos.append(a)
    return Corrida("al azar", np.array(brazos), np.array(recompensas), float("nan"))


def _turnos(b, n, s) -> Corrida:
    rng = np.random.default_rng(s)
    brazos, recompensas = [], []
    for t in range(n):
        a = t % b.n_brazos
        recompensas.append(b.tirar(a, rng))
        brazos.append(a)
    return Corrida("por turnos", np.array(brazos), np.array(recompensas), float("nan"))


def parte4() -> None:
    print("\n  4 · Meridiano como bandido\n")
    print("  Un brazo por habilidad, sin mirar el estado. 500 rondas, 10 semillas.")
    print("  Se mide la ganancia de dominio, que es lo que de verdad importa.\n")
    modelo = ajustar_pfa()

    candidatos = [
        ("ávida pura", lambda b, n, s: epsilon_avida(b, n, epsilon=0.0, seed=s)),
        ("ε-ávida 0,10", lambda b, n, s: epsilon_avida(b, n, epsilon=0.10, seed=s)),
        ("ε-ávida 0,50", lambda b, n, s: epsilon_avida(b, n, epsilon=0.50, seed=s)),
        ("por turnos", _turnos),
        ("al azar", _azar),
        ("UCB1 c=2", lambda b, n, s: ucb1(b, n, c=2.0, seed=s)),
    ]
    print(f"  {'':<16s} {'ganancia':>10s}  {'IC 95 %':>20s}  {'brazos':>8s}")
    resultados = {}
    for nombre, hacer in candidatos:
        ganancias, usados = [], []
        for s in range(10):
            b = MeridianoBandido(modelo, catalogo=20, rondas=500, recompensa="dominio")
            b.reset(seed=s)
            antes = b.env.dominio()
            c = hacer(b, 500, s)
            ganancias.append(b.env.dominio() - antes)
            usados.append(int((c.reparto(20) > 0).sum()))
        m, lo, hi = ic(ganancias)
        resultados[nombre] = (m, lo, hi)
        print(f"  {nombre:<16s} {m:>+10.4f}  [{lo:+7.4f}, {hi:+7.4f}]  "
              f"{np.mean(usados):>6.1f}/20")

    avida = resultados["ávida pura"]
    azar = resultados["al azar"]
    ucb = resultados["UCB1 c=2"]
    print(f"\n  La avida pura se queda en {avida[0]:+.4f}, {azar[0] / max(avida[0], 1e-9):.1f} veces por")
    print("  debajo de tirar al azar.")
    print("  No es que explore poco: es que en cuanto un brazo le da algo, deja de")
    print("  mirar los demas, y ese brazo se AGOTA porque el estudiante ya lo domina.")
    solapan = not (azar[2] < ucb[1] or ucb[2] < azar[1])
    if solapan:
        print(f"\n  Y ahora lo incomodo: UCB1 saca {ucb[0]:+.4f} y tirar AL AZAR saca "
              f"{azar[0]:+.4f}.")
        print("  Sus intervalos se solapan, asi que con estas semillas no se puede")
        print("  afirmar que UCB1 sea mejor que el azar en este problema.")
    print("\n  Aqui el bandido estacionario se rompe: tirar de un brazo cambia ese")
    print("  brazo. Cualquier cosa que reparta las tiradas gana; la sofisticacion")
    print("  aporta poco. Es la misma leccion del proyecto: mida su linea base tonta")
    print("  ANTES de construir encima.\n")


PARTES = {1: parte1, 2: parte2, 3: parte3, 4: parte4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parte", type=int, choices=sorted(PARTES), default=None)
    args = ap.parse_args()
    print("\n  UPTC · Sesion 5 · Bandidos: la decision repetida")
    for n in ([args.parte] if args.parte else sorted(PARTES)):
        PARTES[n]()


if __name__ == "__main__":
    main()
