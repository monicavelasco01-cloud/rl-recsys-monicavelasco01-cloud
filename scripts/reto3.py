"""Arnes del reto 3: entrena con SU recompensa y la mide con la de verdad.

    uv run python scripts/reto3.py

El reto no es "haga que aprenda mas rapido". Es mas dificil que eso: **anada
una pista sin estropear el problema**. Va a descubrir que es sorprendentemente
facil estropearlo.

El arnes hace cuatro comprobaciones y las cuatro tienen que pasar.

1. Su funcion tiene que decir algo. La funcion que devuelve cero siempre no
   cuenta: eso no es una pista, es no contestar.
2. El agente tiene que llegar a la meta al menos tan a menudo como el agente
   que no recibio ninguna pista.
3. Medido con la recompensa **original**, tiene que sacar lo que saca la
   politica optima.
4. Y no puede creerse mucho mejor de lo que es: lo que marca su recompensa no
   puede dispararse por encima de lo que el agente saca de verdad.

La cuarta es la que separa. Un agente no puede saber que la recompensa estaba
mal escrita: hizo exactamente lo que se le pidio. El que se equivoco fue quien
la escribio.

Un aviso para que no se lleve una idea falsa: el moldeado bien hecho **no
conserva el retorno**, conserva cual es la mejor politica. Es normal, y hasta
frecuente, que su agente marque un poco menos que el retorno real. Lo que no
puede es marcar mucho mas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ``retos`` vive en la raiz del repositorio y no es un paquete instalado, asi
# que hay que anadirla a mano. Es la unica linea de fontaneria de este archivo.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from retos.reto3 import mi_moldeado  # noqa: E402

from rlrs.dp import value_iteration  # noqa: E402
from rlrs.envs import ARROWS, sala_escasa  # noqa: E402
from rlrs.shaping import GridWorldConMoldeado, retorno_verdadero  # noqa: E402
from rlrs.td import q_learning  # noqa: E402

EPISODIOS = 600
GAMMA = 0.99
ALPHA = 0.1
SEMILLAS = range(5)

# Los listones. Salen de medir, no de opinar. La linea base sin ninguna pista
# llega a la meta el 100 % de las veces y saca +0,8093; la politica optima saca
# +0,8135. Cualquiera de esas dos cifras es alcanzable.
MINIMO_LLEGADAS = 0.90      # proporcion de episodios que terminan en la meta
MINIMO_RETORNO = 0.78       # retorno real, con la recompensa original
MAXIMO_ENGANO = 0.50        # cuanto puede pasarse por arriba, no por abajo


def entrena(funcion, semilla: int):
    """Entrena una vez y devuelve la politica, el retorno que cree y las llegadas."""
    env = GridWorldConMoldeado(sala_escasa(), funcion)
    llegadas: list[bool] = []
    paso_original = env.step

    def step(a):
        s2, r, term, trunc, info = paso_original(a)
        if term:
            llegadas.append(True)
        elif trunc:
            llegadas.append(False)
        return s2, r, term, trunc, info

    env.step = step  # type: ignore[method-assign]
    ap = q_learning(env, episodes=EPISODIOS, gamma=GAMMA, alpha=ALPHA, seed=semilla)
    tasa = float(np.mean(llegadas[-100:])) if llegadas else 0.0

    # "Lo que cree" es la MISMA politica avida medida en el entorno con su
    # pista dentro. Comparar el entrenamiento contra la evaluacion seria
    # trampa: la diferencia saldria de la exploracion y no de la recompensa.
    cree, _, _ = retorno_verdadero(
        GridWorldConMoldeado(sala_escasa(), funcion), ap.politica,
        episodios=100, max_pasos=300,
    )
    return ap.politica, cree, tasa


def mapa(env, politica: np.ndarray) -> str:
    filas = []
    for f in range(env.n_rows):
        linea = ""
        for c in range(env.n_cols):
            pos = (f, c)
            if env.is_terminal(pos):
                linea += " G"
            else:
                linea += " " + ARROWS[int(politica[env.state_index(pos)])]
        filas.append(linea)
    return "\n".join("    " + f for f in filas)


def es_la_funcion_nula() -> bool:
    """Comprueba que la funcion diga algo, sin entrenar nada."""
    muestras = [
        ((7, 0), (6, 0), False),
        ((6, 0), (6, 1), False),
        ((0, 10), (0, 11), True),
        ((3, 5), (3, 6), False),
        ((1, 9), (2, 9), False),
    ]
    return all(abs(float(mi_moldeado(a, b, t))) < 1e-12 for a, b, t in muestras)


def main() -> int:
    print("\n  Reto 3 · Anada una pista sin estropear el problema\n")

    env = sala_escasa()
    _, politica_optima, _ = value_iteration(env, gamma=GAMMA)
    opt, opt_lo, opt_hi = retorno_verdadero(sala_escasa(), politica_optima, max_pasos=300)

    if es_la_funcion_nula():
        print("  Su funcion devuelve cero en todas las transiciones que probe.")
        print("  Eso no es una pista: es no contestar. Escriba algo en 'mi_moldeado'")
        print("  y vuelva a ejecutar.\n")
        return 1

    crees, reales, tasas, politicas = [], [], [], []
    for semilla in SEMILLAS:
        politica, cree, tasa = entrena(mi_moldeado, semilla)
        real, _, _ = retorno_verdadero(sala_escasa(), politica, episodios=100, max_pasos=300)
        crees.append(cree)
        reales.append(real)
        tasas.append(tasa)
        politicas.append(politica)

    cree = float(np.mean(crees))
    real = float(np.mean(reales))
    margen = 1.96 * float(np.std(reales, ddof=1)) / np.sqrt(len(reales))
    tasa = float(np.mean(tasas))
    engano = cree - real

    print(f"  Cinco semillas, {EPISODIOS} episodios cada una.\n")
    print(f"  Llega a la meta                    {tasa * 100:>8.0f} %      hace falta >= {MINIMO_LLEGADAS * 100:.0f} %")
    print(f"  Lo que su agente CREE que saco     {cree:>+9.4f}      (misma politica, con su pista dentro)")
    print(f"  Lo que de verdad saco              {real:>+9.4f} +-{margen:.4f}   hace falta >= {MINIMO_RETORNO:+.2f}")
    print(f"  Se cree mejor en                   {engano:>+9.4f}      no puede pasar de {MAXIMO_ENGANO:+.2f}")
    print(f"  Referencia, la politica optima     {opt:>+9.4f}      [{opt_lo:+.4f}, {opt_hi:+.4f}]\n")
    print("  Politica aprendida con su pista:")
    print(mapa(env, politicas[0]))

    fallos = []
    if tasa < MINIMO_LLEGADAS:
        fallos.append(
            f"Solo llega a la meta el {tasa * 100:.0f} % de las veces. Sin ninguna pista llegaba "
            "el 100 %.\n     Su pista le esta dando al agente una razon para no llegar."
        )
    if real < MINIMO_RETORNO:
        fallos.append(
            f"El retorno real ({real:+.4f}) no alcanza el de la politica optima.\n"
            "     El agente esta resolviendo un problema parecido, pero no el nuestro."
        )
    if engano > MAXIMO_ENGANO:
        fallos.append(
            f"Su agente se cree {engano:+.4f} mejor de lo que es.\n"
            "     Esa diferencia es lo que le esta pagando su pista por algo que no era el objetivo."
        )

    print()
    if not fallos:
        print("  SUPERADO.  Su pista informa sin cambiar cual es la mejor politica.")
        print("             Compruebe en tests/test_reto3.py que propiedad tiene su")
        print("             funcion, porque no es casualidad.\n")
        return 0

    print("  NO SUPERADO\n")
    for i, f in enumerate(fallos, 1):
        print(f"  {i}. {f}")
    print("\n  No lo arregle todavia si es la primera vez. Anote en la bitacora que")
    print("  escribio, que esperaba y que paso. De eso trata la sesion.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
