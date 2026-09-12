"""Recomendacion, medida.

    uv run python experiments/recomendacion.py             las cuatro partes
    uv run python experiments/recomendacion.py --parte 2   solo una

Cuatro partes:

  1. Los datos, y el sesgo que traen.
  2. **La particion decide el resultado.** El mismo modelo, dos particiones.
  3. Las lineas base, y cual gana.
  4. Las cuatro cifras juntas, y por que ninguna sola vale.

Todos los numeros de la guia del sabado salen de aqui.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlrs.recomendacion import (
    al_azar,
    cargar,
    evaluar,
    factorizacion_implicita,
    knn_items,
    novedad_pura,
    particion_aleatoria,
    particion_temporal,
    por_popularidad,
    repetir_lo_propio,
)

K = 10


def _datos():
    print("  Cargando ASSISTments...", flush=True)
    return cargar()


# ── 1 ───────────────────────────────────────────────────────────────────────
def parte1() -> None:
    print("\n  1 · Los datos, y el sesgo que traen\n")
    d = _datos()
    print(f"     {d.n_usuarios:,} estudiantes con 10 o mas interacciones")
    print(f"     {d.n_items} habilidades en el catalogo")
    print(f"     {d.n_interacciones:,} interacciones\n")

    pop = d.popularidad()
    orden = np.argsort(pop)[::-1]
    print("     Las seis habilidades mas practicadas:\n")
    for k in orden[:6]:
        print(f"       {int(pop[k]):>7,}  {d.nombres.get(int(k), '?')[:52]}")
    top10 = pop[orden[:10]].sum() / pop.sum()
    print(f"\n     Las diez mas practicadas concentran el {100 * top10:.1f} % de todo.")
    print("     Esa concentracion NO dice que esas habilidades gusten mas. Dice que")
    print("     el tutor de entonces las puso mas. La realimentacion es implicita y")
    print("     el sesgo de exposicion viene dentro: los datos no cuentan que quiere")
    print("     cada estudiante, cuentan que le pusieron delante.\n")


# ── 2 ───────────────────────────────────────────────────────────────────────
def parte2() -> None:
    print("\n  2 · La particion decide el resultado\n")
    d = _datos()
    print("     El mismo modelo, los mismos datos, dos formas de partir.\n")

    filas = {}
    for hacer in (particion_temporal, particion_aleatoria):
        p = hacer(d)
        filas[p.nombre] = {}
        print(f"     particion {p.nombre}")
        for constructor in (por_popularidad, repetir_lo_propio, knn_items):
            r = evaluar(p, constructor(p), k=K)
            filas[p.nombre][r.nombre] = r
            print(f"       {r}")
        print()

    # La conclusion se calcula de la tabla, no se escribe a mano.
    t = filas["temporal"]["repetir lo propio"]
    a = filas["aleatoria"]["repetir lo propio"]
    infla = 100 * (a.ndcg - t.ndcg) / t.ndcg
    print(f"     Fijate en «repetir lo propio», que es una linea base de dos lineas.")
    print(f"     Con particion temporal saca nDCG@{K} = {t.ndcg:.4f}.")
    print(f"     Con particion aleatoria saca {a.ndcg:.4f}.")
    print(f"     La misma linea base, {infla:.0f} % mas alta, sin haber cambiado nada.\n")
    print("     Eso es fuga de datos. Al barajar, el modelo ve interacciones POSTERIORES")
    print("     a las que tiene que predecir, y como la gente repite, adivinar el pasado")
    print("     conociendo el futuro es facil. En produccion ese futuro no existe.")
    print("\n     Regla: si la particion no respeta el tiempo, la cifra no significa nada.\n")


# ── 3 ───────────────────────────────────────────────────────────────────────
def parte3() -> None:
    print("\n  3 · Las lineas base, y cual gana\n")
    d = _datos()
    p = particion_temporal(d)
    print(f"     Particion temporal, {p.usuarios_evaluables:,} usuarios evaluables.\n")

    resultados = []
    for constructor in (al_azar, novedad_pura, por_popularidad, repetir_lo_propio):
        resultados.append(evaluar(p, constructor(p), k=K))
    resultados.append(evaluar(p, knn_items(p, vecinos=20), k=K))
    print("     Cargando la factorizacion, unos segundos...", flush=True)
    resultados.append(evaluar(p, factorizacion_implicita(p, factores=32, pasos=10), k=K))

    for r in sorted(resultados, key=lambda x: -x.recall):
        print(f"       {r}")

    mejor = max(resultados, key=lambda r: r.recall)
    repetir = next(r for r in resultados if r.nombre == "repetir lo propio")
    knn = next(r for r in resultados if r.nombre.startswith("kNN"))
    nueva = next(r for r in resultados if r.nombre == "novedad pura")
    azar = next(r for r in resultados if r.nombre == "al azar")

    print(f"\n     Gana {mejor.nombre}, con Recall@{K} = {mejor.recall:.4f}.")
    if knn.recall < repetir.recall:
        print(f"     El filtro colaborativo ({knn.recall:.4f}) pierde contra una linea base")
        print(f"     de dos lineas ({repetir.recall:.4f}). No es raro: es lo habitual, y es")
        print("     el motivo de que este bloque exista.")
    if nueva.recall < azar.recall:
        print(f"\n     Y mira esto: «novedad pura», que solo recomienda cosas que el usuario")
        print(f"     NO ha hecho, saca {nueva.recall:.4f}, por debajo de recomendar AL AZAR")
        print(f"     ({azar.recall:.4f}). La intuicion de que un recomendador no debe repetir")
        print("     es, en estos datos, el error mas caro que se puede cometer.\n")


# ── 4 ───────────────────────────────────────────────────────────────────────
def parte4() -> None:
    print("\n  4 · Las cuatro cifras juntas\n")
    d = _datos()
    p = particion_temporal(d)

    pop = evaluar(p, por_popularidad(p), k=K)
    rep = evaluar(p, repetir_lo_propio(p), k=K)
    print("     Cargando la factorizacion...", flush=True)
    fac = evaluar(p, factorizacion_implicita(p, factores=32, pasos=10), k=K)

    print(f"\n     {'':<26s} {'Recall':>8s} {'nDCG':>8s} {'cobertura':>11s} {'novedad':>9s}")
    for r in (pop, rep, fac):
        print(f"     {r.nombre:<26s} {r.recall:>8.4f} {r.ndcg:>8.4f} "
              f"{r.cobertura:>11.3f} {r.novedad:>9.2f}")

    print(f"\n     Tres lecturas, y las tres hacen falta.\n")
    print(f"     Popularidad saca un Recall respetable ({pop.recall:.4f}) con una cobertura")
    print(f"     de {pop.cobertura:.3f}: recomienda {int(round(pop.cobertura * p.n_items))} items")
    print(f"     de {p.n_items} y nunca ensena nada mas. Un sistema asi encoge el catalogo")
    print("     y nadie se entera mirando el Recall.")
    if fac.recall > rep.recall and fac.ndcg < rep.ndcg:
        print(f"\n     La factorizacion GANA en Recall ({fac.recall:.4f} frente a {rep.recall:.4f})")
        print(f"     y PIERDE en nDCG ({fac.ndcg:.4f} frente a {rep.ndcg:.4f}). Encuentra mas")
        print("     items relevantes y los ordena peor. Con una sola cifra, cualquiera de")
        print("     los dos puede declararse ganador, y los dos tendrian razon.")
    print("\n     Por eso se reportan las cuatro. Una sola cifra no es un resultado:")
    print("     es la cifra que mejor le quedaba a quien escribio el informe.\n")


PARTES = {1: parte1, 2: parte2, 3: parte3, 4: parte4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parte", type=int, choices=sorted(PARTES), default=None)
    args = ap.parse_args()
    print("\n  UPTC · Sesion 6 · Recomendacion: el problema, los datos y el juez")
    for n in ([args.parte] if args.parte else sorted(PARTES)):
        PARTES[n]()


if __name__ == "__main__":
    main()
