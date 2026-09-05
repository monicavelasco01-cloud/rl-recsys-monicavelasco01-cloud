"""Sesion 4. Cuando la tabla se cambia por una red.

    uv run python experiments/profundo.py            todo, unos 4 minutos
    uv run python experiments/profundo.py --parte 2  solo una parte

Cuatro partes:

  1. DQN funciona. Q-learning con una red, sobre la misma cuadricula de la
     sesion 1, comparado con la solucion exacta que ya conocemos.
  2. Las dos muletas de DQN, medidas. Repeticion de experiencia y red
     objetivo, quitandolas de una en una y con varias semillas.
  3. Gradiente de politica. REINFORCE, con linea base y con critico.
  4. El colapso. El modo de fallo que no tienen los metodos de valor.

Todos los numeros de la guia del sabado salen de aqui. Si algo no coincide,
manda la salida entera al foro: puede ser un error del material.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlrs.aprox import caracteristicas_posicion, caracteristicas_tabulares
from rlrs.dp import value_iteration
from rlrs.dqn import ablacion_dqn, dqn
from rlrs.envs import ARROWS, GridWorld, sala_escasa
from rlrs.pg import actor_critico, reinforce
from rlrs.shaping import retorno_verdadero

SEMILLAS = 6
GAMMA = 0.9


def rejilla() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


def ic(xs) -> tuple[float, float, float]:
    a = np.asarray(xs, dtype=float)
    h = 1.96 * float(a.std(ddof=1)) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(a.mean()) - h, float(a.mean()) + h


def mapa(env: GridWorld, politica: np.ndarray) -> str:
    filas = []
    for f in range(env.n_rows):
        linea = ""
        for c in range(env.n_cols):
            pos = (f, c)
            if env.is_wall(pos):
                linea += " # "
            elif env.is_terminal(pos):
                linea += " G " if env.terminals[pos] >= 0 else " X "
            else:
                linea += " " + ARROWS[int(politica[env.state_index(pos)])] + " "
        filas.append(linea)
    return "\n".join("     " + f for f in filas)


# ── 1 ───────────────────────────────────────────────────────────────────────
def parte1() -> None:
    print("\n  1 · DQN sobre la cuadricula de la sesion 1\n")
    valores, politica_optima, barridos = value_iteration(rejilla(), gamma=GAMMA)
    opt, lo, hi = retorno_verdadero(rejilla(), politica_optima)
    print(f"  La solucion exacta, de la sesion 1: {barridos} barridos, retorno "
          f"{opt:+.4f}  [{lo:+.4f}, {hi:+.4f}]\n")

    env = rejilla()
    phi = caracteristicas_posicion(env)
    ap = dqn(env, phi, episodes=300, gamma=GAMMA, seed=0)
    politica = ap.politica(rejilla(), caracteristicas_posicion(rejilla()))
    real, rlo, rhi = retorno_verdadero(rejilla(), politica)
    iguales = int((politica == politica_optima).sum())

    print(f"  DQN, 300 episodios, semilla 0")
    print(f"     retorno real          {real:+.4f}  [{rlo:+.4f}, {rhi:+.4f}]")
    print(f"     coincide con la optima en {iguales} de {politica.size} casillas")
    pesos = sum(p.size for p in ap.red.parametros())
    casillas = rejilla().n_states * 4
    print(f"     pesos de la red: {pesos} numeros, frente a {casillas} casillas de tabla\n")
    print(f"  Si, la red gasta {pesos} numeros para un problema que la tabla resuelve")
    print(f"  con {casillas}. Aqui es un despilfarro y conviene decirlo en voz alta.")
    print("  Lo que cambia no es el tamano, es COMO CRECE cada cosa: la tabla crece")
    print("  con el numero de estados y la red no. En Meridiano hay 325.000")
    print("  interacciones; la tabla no cabe y la red seguiria teniendo 580 numeros.\n")
    print(mapa(rejilla(), politica))
    print("\n     (la optima, para comparar)")
    print(mapa(rejilla(), politica_optima))
    print("\n  No es una tabla disfrazada: la red recibe cuatro numeros por estado")
    print("  y de ahi saca los cuatro valores. Nunca ve el indice de la casilla.\n")


# ── 2 ───────────────────────────────────────────────────────────────────────
def parte2() -> None:
    print("\n  2 · Las dos muletas de DQN, medidas\n")
    print("  Repeticion de experiencia y red objetivo son la respuesta clasica a la")
    print("  triada de ayer. Aqui se quitan de una en una, con 6 semillas cada una.\n")

    _, politica_optima, _ = value_iteration(rejilla(), gamma=GAMMA)
    opt, _, _ = retorno_verdadero(rejilla(), politica_optima)

    combinaciones = [
        (True, True, "DQN completo"),
        (True, False, "sin red objetivo"),
        (False, True, "sin memoria"),
        (False, False, "sin ninguna de las dos"),
    ]
    print(f"  {'':<24s} {'retorno real':>14s}  {'IC 95 %':>22s}  {'peor':>9s}")
    resultados = {}
    for con_mem, con_obj, etiqueta in combinaciones:
        reales = []
        for semilla in range(SEMILLAS):
            env = rejilla()
            ap = dqn(env, caracteristicas_posicion(env), episodes=300, gamma=GAMMA,
                     seed=semilla, usar_memoria=con_mem, usar_red_objetivo=con_obj)
            pol = ap.politica(rejilla(), caracteristicas_posicion(rejilla()))
            reales.append(retorno_verdadero(rejilla(), pol, episodios=100)[0])
        m, lo, hi = ic(reales)
        resultados[etiqueta] = (m, lo, hi, min(reales))
        print(f"  {etiqueta:<24s} {m:>+14.4f}  [{lo:+.4f}, {hi:+.4f}]  {min(reales):>+9.4f}")

    print(f"\n  Referencia: la politica optima saca {opt:+.4f}.\n")

    # La conclusion se calcula a partir de la tabla, no se escribe a mano. Asi
    # no puede contradecir a sus propios numeros, que es un fallo mas comun de
    # lo que parece en informes de este tipo.
    orden = sorted(resultados.items(), key=lambda kv: -kv[1][0])
    mejor, peor = orden[0], orden[-1]
    completo = resultados["DQN completo"]
    print(f"  Lo mejor de esta tanda: {mejor[0]} ({mejor[1][0]:+.4f}).")
    print(f"  Lo peor:                {peor[0]} ({peor[1][0]:+.4f}).")
    if completo[0] < mejor[1][0]:
        print(f"  DQN completo NO gana: queda {mejor[1][0] - completo[0]:+.4f} por debajo.")
    solapan = [e for e, (m, lo, hi, _) in resultados.items()
               if e != mejor[0] and hi >= mejor[1][1]]
    if solapan:
        print(f"\n  Ojo antes de sacar conclusiones: el intervalo de {mejor[0]} se solapa")
        print(f"  con el de {', '.join(solapan)}. Con {SEMILLAS} semillas NO se puede")
        print("  afirmar cual es mejor entre esos. Cambia SEMILLAS y veras que el orden")
        print("  de los primeros baila. Ese baile es, por si solo, el resultado.")

    print("\n  Lo que si aguanta el cambio de semillas es esto: la red objetivo SIN")
    print("  memoria es la peor combinacion de las cuatro. Una muleta sin la otra")
    print("  estorba, porque el objetivo congelado retrasa la propagacion y sin")
    print("  memoria no hay lotes que compensen el retraso.")
    print("\n  Y la conclusion util no es 'no usar DQN'. Es que este problema es")
    print("  demasiado facil para necesitarlo, y que una tecnica se elige midiendo")
    print("  en TU problema, no copiando la del articulo.\n")


# ── 3 ───────────────────────────────────────────────────────────────────────
def parte3() -> None:
    print("\n  3 · Gradiente de politica\n")
    print("  Aqui no hay tabla ni argmax: la red devuelve una probabilidad por")
    print("  accion, y se ajusta para que suba el retorno esperado.\n")

    metodos = [
        ("REINFORCE", reinforce, {"lr": 5e-3}),
        ("REINFORCE + linea base", reinforce, {"lr": 5e-3, "linea_base": True}),
        ("actor-critico", actor_critico, {}),
    ]
    print(f"  {'':<24s} {'retorno real':>14s}  {'IC 95 %':>22s}  {'colapsos':>9s}")
    for etiqueta, fn, kw in metodos:
        reales = []
        for semilla in range(SEMILLAS):
            env = rejilla()
            ap = fn(env, caracteristicas_posicion(env), episodes=1200, gamma=GAMMA,
                    seed=semilla, **kw)
            pol = ap.politica(rejilla(), caracteristicas_posicion(rejilla()))
            reales.append(retorno_verdadero(rejilla(), pol, episodios=100)[0])
        m, lo, hi = ic(reales)
        colapsos = sum(1 for r in reales if r < -1.0)
        print(f"  {etiqueta:<24s} {m:>+14.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{colapsos:>4d}/{SEMILLAS}")

    print("\n  Otra vez lo mismo, y otra vez al reves de lo que dice el manual: la")
    print("  version mas simple es la unica que no falla nunca. La linea base y el")
    print("  critico bajan una varianza que en un problema de veinte casillas casi")
    print("  no existe, y a cambio introducen ventajas negativas que empujan a la")
    print("  politica a volverse determinista. Cuando eso pasa, pasa lo de la parte 4.\n")


# ── 4 ───────────────────────────────────────────────────────────────────────
def parte4() -> None:
    print("\n  4 · El colapso: el modo de fallo que los metodos de valor no tienen\n")
    env = rejilla()
    ap = actor_critico(env, caracteristicas_posicion(env), episodes=1200,
                       gamma=GAMMA, seed=0)
    pol = ap.politica(rejilla(), caracteristicas_posicion(rejilla()))
    real, _, _ = retorno_verdadero(rejilla(), pol, episodios=100)
    r = ap.retornos

    print("  Actor-critico, semilla 0. Por tramos de episodios:\n")
    print(f"     {'tramo':<16s} {'retorno medio':>14s} {'desviacion':>12s} {'llegan a la meta':>18s}")
    for a, b in [(0, 100), (200, 300), (500, 600), (900, 1000), (1100, 1200)]:
        tramo = r[a:b]
        # En esta rejilla, un episodio que se agota da exactamente -4,0000.
        llegan = float(np.mean(tramo > -3.99)) * 100.0
        print(f"     {f'{a} a {b}':<16s} {float(tramo.mean()):>+14.4f} "
              f"{float(tramo.std()):>12.4f} {llegan:>17.0f} %")

    from rlrs.pg import softmax
    X = np.stack([caracteristicas_posicion(rejilla())(s) for s in range(env.n_states)])
    p = softmax(ap.red.adelante(X))
    concentracion = float(p.max(axis=1).mean())
    azar = 1.0 / env.n_actions

    print(f"\n  Retorno de la politica final, evaluada: {real:+.4f}")
    print(f"  Probabilidad de la accion mas probable: {concentracion:.4f} "
          f"(el azar puro daria {azar:.2f})")

    print("\n  El diagnostico esta en la columna de la desviacion. Cuando llega a")
    print("  cero, todos los episodios dan exactamente lo mismo, y un metodo que")
    print("  aprende comparando episodios entre si se queda sin nada que comparar.")
    print("  La ventaja se hace cero, el gradiente se hace cero, y ahi acaba todo.")
    print("\n  Fijate en que NO es que aprenda despacio: es que ya no hay senal. Por")
    print("  eso no se arregla con mas episodios ni con otra tasa de aprendizaje.")
    print("  Un metodo de valor con epsilon-avida no puede caer aqui, porque sigue")
    print("  probando acciones aunque su politica actual sea mala.")
    print("\n  Arreglarlo es el reto de hoy: uv run python scripts/reto4.py\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parte", type=int, choices=[1, 2, 3, 4])
    args = ap.parse_args()
    print("\n  UPTC · Sesion 4 · Cuando la tabla se cambia por una red")
    partes = {1: parte1, 2: parte2, 3: parte3, 4: parte4}
    for n in ([args.parte] if args.parte else [1, 2, 3, 4]):
        partes[n]()


if __name__ == "__main__":
    main()
