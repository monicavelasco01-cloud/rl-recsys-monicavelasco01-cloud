"""Taller C · Las lineas base, y lo que el Recall no ve.

    uv run python talleres/taller_c_lineas_base.py

Lo dirigen los grupos 1 y 2. Tres pasos:

  C1. Que catalogo recomienda de verdad cada modelo. La popularidad acierta el
      41 % y el curso entero descubre cuantos items distintos ha llegado a
      nombrar.
  C2. La linea base que humilla al filtro colaborativo, y por que gana.
  C3. La trampa contraria: «novedad pura» por debajo del azar. La intuicion de
      que un recomendador no debe repetir, medida.
"""

from __future__ import annotations

import numpy as np

from rlrs.recomendacion import (
    al_azar,
    cargar,
    evaluar,
    factorizacion_implicita,
    knn_items,
    novedad_pura,
    particion_temporal,
    por_popularidad,
    repetir_lo_propio,
)

K = 10

CORTO = {
    "factorización (32 factores)": "factorización",
    "kNN por ítems (20)": "kNN por ítems",
}
corto = lambda n: CORTO.get(n, n)  # noqa: E731


# ── C1 ──────────────────────────────────────────────────────────────────────
def c1(d, p) -> dict:
    print("\n  C1 · Que catalogo recomienda de verdad\n")
    print("     Cargando la factorizacion, unos segundos...", flush=True)
    res = {}
    for constructor in (al_azar, novedad_pura, por_popularidad, repetir_lo_propio,
                        knn_items,
                        lambda q: factorizacion_implicita(q, factores=32, pasos=10)):
        r = evaluar(p, constructor(p), k=K)
        res[r.nombre] = r

    print(f"\n       {'':<18s} {'Recall':>8s} {'nDCG':>8s} {'cobertura':>11s} "
          f"{'ítems':>7s} {'novedad':>9s}")
    for r in sorted(res.values(), key=lambda x: -x.recall):
        items = int(round(r.cobertura * p.n_items))
        print(f"       {corto(r.nombre):<18s} {r.recall:>8.4f} {r.ndcg:>8.4f} "
              f"{r.cobertura:>11.3f} {items:>7d} {r.novedad:>9.2f}")

    pop = res["popularidad"]
    n_pop = int(round(pop.cobertura * p.n_items))
    print(f"\n     Miren la fila de la popularidad. Acierta el {100 * pop.recall:.0f} %, mas que")
    print(f"     el kNN y mas que la novedad, y en toda la evaluacion ha llegado a")
    print(f"     nombrar {n_pop} habilidades distintas de {p.n_items}. Las otras {p.n_items - n_pop}")
    print("     no existen para ese sistema, y el Recall no lo dice.\n")

    pobl = d.popularidad()
    orden = np.argsort(pobl)[::-1][:n_pop]
    print(f"     Las {n_pop} unicas que recomienda, en orden:\n")
    for k in orden:
        print(f"       {int(pobl[k]):>7,}  {d.nombres.get(int(k), '?')[:56]}")
    print("\n     Esa es la lista completa. Un estudiante que use ese sistema durante")
    print("     todo el semestre no vera nunca nada fuera de estas.\n")
    return res


# ── C2 ──────────────────────────────────────────────────────────────────────
def c2(p, res) -> None:
    print("\n  C2 · Dos lineas de codigo contra un filtro colaborativo\n")
    rep = res["repetir lo propio"]
    knn = res["kNN por ítems (20)"]

    print("     «Repetir lo propio» es literalmente esto:\n")
    print("       los items que este usuario ya practico, el mas repetido primero,")
    print("       y despues los mas populares para rellenar\n")
    print(f"       repetir lo propio   Recall@{K} = {rep.recall:.4f}")
    print(f"       kNN por items       Recall@{K} = {knn.recall:.4f}")
    print(f"\n     {rep.recall / knn.recall:.2f} veces mejor, sin aprender nada.\n")

    # Por que gana: cuanto de lo que viene es repeticion.
    repes, total = 0, 0
    for tr, te in zip(p.train, p.test):
        if len(te) == 0:
            continue
        conocidos = set(tr.tolist())
        repes += sum(1 for i in te.tolist() if i in conocidos)
        total += len(te)
    print(f"     El motivo esta en los datos, no en el algoritmo: el "
          f"{100 * repes / total:.1f} % de lo")
    print("     que un estudiante practica manana ya lo practico antes. En un dominio")
    print("     asi, «lo de siempre» es una prediccion excelente.")
    print("\n     Pregunta para el curso, y es la importante del taller: en que")
    print("     dominios NO se cumple esto, y que le pasaria alli a esta linea base.\n")


# ── C3 ──────────────────────────────────────────────────────────────────────
def c3(res) -> None:
    print("\n  C3 · La intuicion mas cara del curso\n")
    nueva = res["novedad pura"]
    azar = res["al azar"]

    print("     «Novedad pura» hace lo que casi todo el mundo diria que hay que")
    print("     hacer: no repetir. Solo recomienda habilidades que el estudiante")
    print("     no ha tocado, empezando por las menos frecuentes.\n")
    print(f"       novedad pura   Recall@{K} = {nueva.recall:.4f}   "
          f"novedad = {nueva.novedad:.2f} bits")
    print(f"       al azar        Recall@{K} = {azar.recall:.4f}   "
          f"novedad = {azar.novedad:.2f} bits")

    if nueva.recall < azar.recall:
        print(f"\n     Queda POR DEBAJO de recomendar al azar, y encima con menos")
        print(f"     novedad medida ({nueva.novedad:.2f} bits frente a {azar.novedad:.2f}).")
        print("     Prohibirse repetir no solo cuesta aciertos: en estos datos ni")
        print("     siquiera compra la sorpresa por la que se pagaron.")

    print("\n     Y aun asi «novedad pura» no es un error de programacion: es una")
    print("     decision de producto, y en un catalogo de peliculas seria la")
    print("     correcta. La leccion no es «hay que repetir»: es que la linea base")
    print("     que hay que batir la dicta el dominio, y se mide antes de elegirla.\n")


def main() -> None:
    print("\n  UPTC · Sesion 6 · Taller C · Las lineas base y la cobertura")
    print("  Lo dirigen los grupos 1 y 2\n")
    print("  Cargando ASSISTments...", flush=True)
    d = cargar()
    p = particion_temporal(d)
    print(f"  {d.n_usuarios:,} estudiantes · {d.n_items} habilidades · "
          f"particion temporal")
    res = c1(d, p)
    c2(p, res)
    c3(res)


if __name__ == "__main__":
    main()
