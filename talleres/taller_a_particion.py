"""Taller A · La particion decide el resultado.

    uv run python talleres/taller_a_particion.py

Lo dirigen los grupos 5 y 6. Tres pasos, y entre paso y paso se para y se
pregunta al resto del curso. Las preguntas estan en la guia del taller; aqui
solo esta lo que hay que ejecutar delante de ellos.

El taller demuestra tres cosas, en este orden:

  A1. Que la particion aleatoria deja pasar informacion del futuro, y cuanta.
  A2. Que eso infla las cifras, y ademas cambia el orden de los modelos.
  A3. Que incluso dentro de la particion correcta, DONDE se corta mueve el
      resultado, asi que el corte se declara junto con la cifra.
"""

from __future__ import annotations

from rlrs.recomendacion import (
    cargar,
    evaluar,
    factorizacion_implicita,
    knn_items,
    particion_aleatoria,
    particion_temporal,
    por_popularidad,
    repetir_lo_propio,
)

K = 10

# Los nombres largos no caben en una tabla proyectada.
CORTO = {
    "factorización (32 factores)": "factorización",
    "kNN por ítems (20)": "kNN por ítems",
}
corto = lambda n: CORTO.get(n, n)  # noqa: E731


def tasa_de_repeticion(particion) -> float:
    """Que fraccion de lo escondido ya estaba a la vista.

    Es la medida directa de la fuga: si lo que hay que predecir ya aparece en
    el entrenamiento, predecirlo no demuestra nada.
    """
    vistos, total = 0, 0
    for tr, te in zip(particion.train, particion.test):
        if len(te) == 0:
            continue
        conocidos = set(tr.tolist())
        vistos += sum(1 for i in te.tolist() if i in conocidos)
        total += len(te)
    return vistos / total if total else 0.0


# ── A1 ──────────────────────────────────────────────────────────────────────
def a1(d) -> tuple:
    print("\n  A1 · Cuanto del futuro se cuela\n")
    tem = particion_temporal(d)
    ale = particion_aleatoria(d)

    print("       particion      lo escondido que YA estaba en el entrenamiento")
    for p in (tem, ale):
        print(f"       {p.nombre:<13s}  {100 * tasa_de_repeticion(p):>5.1f} %")

    print("\n     Las dos esconden el 20 % de cada historial. La diferencia es")
    print("     CUAL 20 %. Al barajar, lo escondido queda rodeado de interacciones")
    print("     posteriores que el modelo si ve, y como la gente repite habilidades,")
    print("     la respuesta esta practicamente escrita en el enunciado.\n")
    return tem, ale


# ── A2 ──────────────────────────────────────────────────────────────────────
def a2(tem, ale) -> None:
    print("\n  A2 · Lo que eso le hace a las cifras\n")
    print("     Cargando la factorizacion en las dos particiones, unos segundos...",
          flush=True)
    filas = {}
    for p in (tem, ale):
        filas[p.nombre] = {}
        for constructor in (por_popularidad, knn_items, repetir_lo_propio,
                            lambda q: factorizacion_implicita(q, factores=32, pasos=10)):
            r = evaluar(p, constructor(p), k=K)
            filas[p.nombre][r.nombre] = r

    print(f"\n       {'':<18s} {'temporal':>17s} {'aleatoria':>19s}")
    print(f"       {'':<18s} {'Recall':>9s}{'nDCG':>8s} {'Recall':>11s}{'nDCG':>8s}")
    for nombre in filas["temporal"]:
        t, a = filas["temporal"][nombre], filas["aleatoria"][nombre]
        print(f"       {corto(nombre):<18s} {t.recall:>9.4f}{t.ndcg:>8.4f} "
              f"{a.recall:>11.4f}{a.ndcg:>8.4f}")

    t = filas["temporal"]["repetir lo propio"]
    a = filas["aleatoria"]["repetir lo propio"]
    infla = 100 * (a.ndcg - t.ndcg) / t.ndcg
    print(f"\n     «Repetir lo propio» sube su nDCG@{K} un {infla:.0f} % sin que se haya")
    print("     tocado una sola linea del recomendador.\n")

    # El orden, que es lo que de verdad duele.
    orden_t = sorted(filas["temporal"].values(), key=lambda r: -r.recall)
    orden_a = sorted(filas["aleatoria"].values(), key=lambda r: -r.recall)
    print("     Orden por Recall, de mejor a peor:\n")
    print(f"       temporal   {' > '.join(corto(r.nombre) for r in orden_t)}")
    print(f"       aleatoria  {' > '.join(corto(r.nombre) for r in orden_a)}")
    if orden_t[0].nombre != orden_a[0].nombre:
        print(f"\n     Fijense en el primer puesto. Con la particion correcta gana")
        print(f"     «{corto(orden_t[0].nombre)}»; al barajar gana «{corto(orden_a[0].nombre)}».")
        print("     El ganador del concurso depende de como se partieron los datos.")
    print("\n     Ese es el dano real. Una cifra inflada se puede descontar; un")
    print("     orden invertido hace que se elija el modelo equivocado.\n")


# ── A3 ──────────────────────────────────────────────────────────────────────
def a3(d) -> None:
    print("\n  A3 · Donde se corta tambien decide\n")
    print("     Misma particion temporal, mismo modelo. Solo cambia la fraccion")
    print("     que se deja ver.\n")
    print(f"       {'fraccion vista':>15s} {'Recall@' + str(K):>12s} {'nDCG@' + str(K):>10s}")
    valores = []
    for fraccion in (0.5, 0.7, 0.8, 0.9):
        p = particion_temporal(d, fraccion=fraccion)
        r = evaluar(p, repetir_lo_propio(p), k=K)
        valores.append((fraccion, r))
        print(f"       {fraccion:>15.0%} {r.recall:>12.4f} {r.ndcg:>10.4f}")

    lo = min(valores, key=lambda v: v[1].recall)
    hi = max(valores, key=lambda v: v[1].recall)
    print(f"\n     El mismo recomendador va de {lo[1].recall:.4f} a {hi[1].recall:.4f} segun")
    print(f"     donde se ponga el corte ({lo[0]:.0%} frente a {hi[0]:.0%}). No es ruido:")
    print("     cuanto mas historial se ve, mas facil es acertar el resto.")
    print("\n     Regla que se lleva el curso: una cifra de recomendacion sin decir")
    print("     que particion y que corte la produjeron no es un resultado.\n")


def main() -> None:
    print("\n  UPTC · Sesion 6 · Taller A · La particion")
    print("  Lo dirigen los grupos 5 y 6\n")
    print("  Cargando ASSISTments...", flush=True)
    d = cargar()
    print(f"  {d.n_usuarios:,} estudiantes · {d.n_items} habilidades · "
          f"{d.n_interacciones:,} interacciones")
    tem, ale = a1(d)
    a2(tem, ale)
    a3(d)


if __name__ == "__main__":
    main()
