"""Los modelos del bloque 4, medidos.

    uv run python experiments/modelos.py             las tres partes
    uv run python experiments/modelos.py --parte 2   solo una

Tres partes:

  1. ALS, BPR y dos torres contra las lineas base, con semillas e intervalo.
  2. **El usuario que llega manana.** Arranque en frio, y la cifra que lo dice.
  3. **El bucle de realimentacion.** El sistema se come sus propios datos.

Todos los numeros de la guia del viernes 18 salen de aqui.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlrs import bucle
from rlrs.recomendacion import (
    al_azar,
    bpr,
    cargar,
    dos_torres,
    evaluar,
    factorizacion_implicita,
    knn_items,
    particion_temporal,
    por_popularidad,
    repetir_lo_propio,
    usuarios_nuevos,
)

K = 10
SEMILLAS = 5


def _ic(v) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def _datos():
    print("  Cargando ASSISTments...", flush=True)
    return cargar()


# ── 1 ───────────────────────────────────────────────────────────────────────
def parte1() -> None:
    print("\n  1 · Tres modelos, y la línea base que sigue ahí\n")
    d = _datos()
    p = particion_temporal(d)
    print(f"     Partición temporal, {p.usuarios_evaluables:,} usuarios, "
          f"{SEMILLAS} semillas.\n")

    base = evaluar(p, repetir_lo_propio(p), k=K)
    print("     Midiendo, medio minuto...\n", flush=True)
    print(f"     {'':<18}{'Recall@' + str(K):>22}{'nDCG@' + str(K):>22}")
    print(f"     {'repetir lo propio':<18}{base.recall:>13.4f}   {'determinista':<7}"
          f"{base.ndcg:>12.4f}   {'determinista':<7}")

    tabla = {}
    for nombre, hacer in (
        ("ALS", lambda q, s: factorizacion_implicita(q, factores=32, pasos=10, seed=s)),
        ("BPR", lambda q, s: bpr(q, seed=s)),
        ("dos torres", lambda q, s: dos_torres(q, seed=s)),
    ):
        rs = [evaluar(p, hacer(p, s), k=K) for s in range(SEMILLAS)]
        r, n = _ic([x.recall for x in rs]), _ic([x.ndcg for x in rs])
        tabla[nombre] = (r, n)
        print(f"     {nombre:<18}{r[0]:>13.4f} ± {r[1]:<6.4f}{n[0]:>12.4f} ± {n[1]:<6.4f}",
              flush=True)

    mejor_r = max(tabla, key=lambda x: tabla[x][0][0])
    a, b = tabla["BPR"][0], tabla["dos torres"][0]
    print(f"\n     Gana {mejor_r} en Recall.")
    if abs(a[0] - b[0]) <= a[1] + b[1]:
        print("     BPR y dos torres **no son distinguibles**: sus intervalos se tocan.")
        print("     Decir eso es parte del resultado, y no decirlo es el error más común")
        print("     de este bloque.")

    ganadores_ndcg = [n for n in tabla if tabla[n][1][0] > base.ndcg]
    if not ganadores_ndcg:
        print(f"\n     Y ahora la fila de arriba. **Repetir lo propio ({base.ndcg:.4f}) ordena")
        print("     mejor que los tres modelos aprendidos.** Tres arquitecturas, cientos de")
        print("     miles de parámetros, y la línea base de dos líneas sigue por delante en")
        print("     nDCG. Encuentran más ítems relevantes y los colocan peor.\n")


# ── 2 ───────────────────────────────────────────────────────────────────────
def parte2() -> None:
    print("\n  2 · El usuario que llega mañana\n")
    d = _datos()
    p = particion_temporal(d)
    frio = usuarios_nuevos(p, fraccion=0.2, seed=0)
    print(f"     Ajuste con {len(frio.ajuste):,} estudiantes. Evaluación con "
          f"{len(frio.train):,} que el modelo NO vio nunca.\n")

    print(f"     {'':<20}{'normal':>20}{'usuarios nuevos':>30}")
    print(f"     {'':<20}{'Recall':>10}{'nDCG':>10}{'Recall':>12}{'nDCG':>10}"
          f"{'sin respuesta':>15}")
    filas = {}
    for hacer in (por_popularidad, repetir_lo_propio,
                  lambda q: knn_items(q, vecinos=20),
                  lambda q: factorizacion_implicita(q, factores=32, pasos=10),
                  bpr, dos_torres):
        a = evaluar(p, hacer(p), k=K)
        b = evaluar(frio, hacer(frio), k=K)
        nombre = a.nombre.split(" (")[0]
        filas[nombre] = (a, b)
        print(f"     {nombre:<20}{a.recall:>10.4f}{a.ndcg:>10.4f}"
              f"{b.recall:>12.4f}{b.ndcg:>10.4f}{100 * b.sin_respuesta:>14.1f} %")

    hunde = [n for n, (a, b) in filas.items() if b.sin_respuesta > 0.5]
    aguanta = [n for n, (a, b) in filas.items()
               if b.sin_respuesta == 0.0 and b.recall > 0.6]
    if hunde:
        peor = max(hunde, key=lambda n: filas[n][1].sin_respuesta)
        a, b = filas[peor]
        print(f"\n     {', '.join(hunde)} pasan de {a.recall:.4f} a {b.recall:.4f}.")
        print(f"     Y la columna de la derecha dice por qué: a un "
              f"{100 * b.sin_respuesta:.0f} % de los usuarios")
        print("     nuevos **no les devuelven nada**. Lista vacía. No es que recomienden")
        print("     mal: es que no tienen un vector para ese usuario, porque un vector por")
        print("     usuario es lo único que aprendieron.")
        print("\n     El resto que sí responde recibe las recomendaciones de otra persona")
        print("     que resultó tener exactamente el mismo historial. Eso también es lo")
        print("     que significa aprender un vector por usuario.")
    if aguanta:
        print(f"\n     {', '.join(aguanta)} aguantan, y por motivos distintos.")
        print("     Popularidad y repetir lo propio no aprenden nada del usuario, así que")
        print("     no tienen nada que perder. **Dos torres sí aprende, y aun así aguanta**,")
        print("     porque calcula el vector del usuario a partir de su historial en vez de")
        print("     buscarlo en una tabla. Esa es toda la diferencia, y es de arquitectura.\n")


# ── 3 ───────────────────────────────────────────────────────────────────────
def parte3(rondas: int = 20, usuarios: int = 300) -> None:
    print("\n  3 · El bucle de realimentación\n")
    d = _datos()
    print(f"     {rondas} rondas, {usuarios} usuarios, registro inicial al 20 %.")
    print("     En cada ronda el recomendador propone, el usuario acepta algo, lo")
    print("     aceptado entra en el registro, y el recomendador se REAJUSTA con el")
    print("     registro ampliado. Nadie toca una línea de código.\n")

    print(f"     {'recomendador':<20}{'catálogo efectivo':>20}{'homogeneidad':>24}"
          f"{'aceptación':>20}")
    print(f"     {'':<20}{'r1':>9}{'r' + str(rondas):>11}{'r1':>12}"
          f"{'r' + str(rondas):>11}{'cambio':>10}{'r1':>10}{'r' + str(rondas):>10}")
    filas = {}
    for nombre, hacer in (("al azar", al_azar), ("popularidad", por_popularidad),
                          ("repetir lo propio", repetir_lo_propio),
                          ("kNN por ítems", lambda q: knn_items(q, vecinos=20)),
                          ("factorización",
                           lambda q: factorizacion_implicita(q, factores=32, pasos=8)),
                          ("dos torres", lambda q: dos_torres(q, pasos=12))):
        f = bucle.simular(d, hacer, rondas=rondas, usuarios=usuarios, k=K,
                          fraccion=0.2, seed=0)
        a, b = f[0], f[-1]
        cambio = 100 * (b.homogeneidad - a.homogeneidad) / a.homogeneidad
        filas[nombre] = (a, b, cambio)
        print(f"     {nombre:<20}{a.catalogo_efectivo:>9.1f}{b.catalogo_efectivo:>11.1f}"
              f"{a.homogeneidad:>12.4f}{b.homogeneidad:>11.4f}{cambio:>9.1f} %"
              f"{a.aceptacion:>10.3f}{b.aceptacion:>10.3f}")

    azar = filas["al azar"][2]
    print(f"\n     La fila de «al azar» sube un {azar:.1f} %, y esa es la deriva natural de")
    print("     añadir datos a un registro. Es la línea de base contra la que se leen")
    print("     todas las demás.")

    peor = max((n for n in filas if n != "al azar"), key=lambda n: filas[n][2])
    print(f"\n     El que más homogeneiza es **{peor}**, con un {filas[peor][2]:.1f} %.")
    knn, fac = filas.get("kNN por ítems"), filas.get("factorización")
    if knn and fac and knn[2] > fac[2]:
        print(f"     Y fíjense en el orden: el kNN ({knn[2]:.1f} %), que es el que aprende de")
        print(f"     los parecidos, homogeneiza más que la factorización ({fac[2]:.1f} %).")
        print("     El método que busca gente parecida es el que más rápido convierte a")
        print("     todo el mundo en la misma persona.")

    suben = [n for n, (a, b, _) in filas.items() if b.aceptacion > a.aceptacion]
    if suben:
        print(f"\n     Y ahora la columna de la derecha: la aceptación **sube** en "
              f"{len(suben)} de {len(filas)}.")
        print("     Esa es la frase de la sesión. **El acierto sube y el catálogo se apaga,")
        print("     a la vez, y nadie tocó una línea de código.** Cualquier panel que mire")
        print("     solo el acierto va a decir que el sistema está mejorando.\n")


PARTES = {1: parte1, 2: parte2, 3: parte3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parte", type=int, choices=sorted(PARTES), default=None)
    args = ap.parse_args()
    print("\n  UPTC · Sesión 7 · Construir el sistema, y perderlo")
    for n in ([args.parte] if args.parte else sorted(PARTES)):
        PARTES[n]()


if __name__ == "__main__":
    main()
