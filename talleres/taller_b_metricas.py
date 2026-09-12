"""Taller B · Una cifra no es un resultado.

    uv run python talleres/taller_b_metricas.py

Lo dirigen los grupos 3 y 4. Tres pasos:

  B1. Recall y nDCG a mano, sobre dos listas de diez que ACIERTAN LO MISMO.
      Aqui no hay datos ni modelos: es aritmetica en la pizarra, y sirve para
      que nadie llegue al paso 2 sin saber que mide cada cifra.
  B2. El mismo par de recomendadores, y cada uno gana una metrica.
  B3. La k tambien decide. El ganador cambia segun cuantos items se muestren.
"""

from __future__ import annotations

import numpy as np

from rlrs.recomendacion import (
    cargar,
    evaluar,
    factorizacion_implicita,
    knn_items,
    ndcg_en_k,
    particion_temporal,
    por_popularidad,
    recall_en_k,
    repetir_lo_propio,
)

K = 10

CORTO = {
    "factorización (32 factores)": "factorización",
    "kNN por ítems (20)": "kNN por ítems",
}
corto = lambda n: CORTO.get(n, n)  # noqa: E731


# ── B1 ──────────────────────────────────────────────────────────────────────
def b1() -> None:
    print("\n  B1 · Los mismos aciertos, en distinto sitio\n")
    relevantes = {7, 42, 99}
    arriba = [7, 42, 99, 1, 2, 3, 4, 5, 6, 8]      # los tres, en los tres primeros
    abajo = [1, 2, 3, 4, 5, 6, 8, 7, 42, 99]       # los tres, en los tres ultimos

    print(f"     El usuario va a practicar estas tres: {sorted(relevantes)}")
    print("     Dos recomendadores devuelven diez items cada uno.\n")
    print(f"       recomendador A: {arriba}")
    print(f"       recomendador B: {abajo}\n")

    print(f"       {'':<16s} {'Recall@10':>10s} {'nDCG@10':>10s}")
    for nombre, lista in (("A (arriba)", arriba), ("B (abajo)", abajo)):
        r = recall_en_k(lista, relevantes, K)
        n = ndcg_en_k(lista, relevantes, K)
        print(f"       {nombre:<16s} {r:>10.4f} {n:>10.4f}")

    print("\n     Los dos aciertan las tres. El Recall no distingue: para el, una")
    print("     lista es un conjunto y el orden no existe. El nDCG si distingue,")
    print("     porque cada acierto vale 1/log2(posicion + 2).\n")

    print("     De donde sale el nDCG de A, termino a termino:\n")
    dcg = 0.0
    for pos, item in enumerate(arriba[:K]):
        if item in relevantes:
            aporte = 1.0 / np.log2(pos + 2)
            dcg += aporte
            print(f"       posicion {pos + 1}: item {item} acierta, aporta "
                  f"1/log2({pos + 2}) = {aporte:.4f}")
    ideal = sum(1.0 / np.log2(p + 2) for p in range(len(relevantes)))
    print(f"       DCG = {dcg:.4f}   ideal = {ideal:.4f}   nDCG = {dcg / ideal:.4f}")
    print("\n     Y el de B, por si alguien cree que el castigo es pequeno:\n")
    dcg_b = 0.0
    for pos, item in enumerate(abajo[:K]):
        if item in relevantes:
            aporte = 1.0 / np.log2(pos + 2)
            dcg_b += aporte
            print(f"       posicion {pos + 1}: item {item} acierta, aporta "
                  f"1/log2({pos + 2}) = {aporte:.4f}")
    print(f"       DCG = {dcg_b:.4f}   ideal = {ideal:.4f}   nDCG = {dcg_b / ideal:.4f}")
    print(f"\n     Mismo Recall, nDCG {dcg / dcg_b:.2f} veces mayor. Si su informe lleva")
    print("     solo Recall, A y B son el mismo sistema.\n")


# ── B2 ──────────────────────────────────────────────────────────────────────
def b2(p) -> dict:
    print("\n  B2 · Cada uno gana una metrica\n")
    print("     Cargando la factorizacion, unos segundos...", flush=True)
    res = {}
    for constructor in (por_popularidad, knn_items, repetir_lo_propio,
                        lambda q: factorizacion_implicita(q, factores=32, pasos=10)):
        r = evaluar(p, constructor(p), k=K)
        res[r.nombre] = r

    print(f"\n       {'':<18s} {'Recall@' + str(K):>11s} {'nDCG@' + str(K):>10s}")
    for r in sorted(res.values(), key=lambda x: -x.recall):
        print(f"       {corto(r.nombre):<18s} {r.recall:>11.4f} {r.ndcg:>10.4f}")

    fac = res["factorización (32 factores)"]
    rep = res["repetir lo propio"]
    if fac.recall > rep.recall and fac.ndcg < rep.ndcg:
        print(f"\n     La factorizacion gana en Recall ({fac.recall:.4f} frente a "
              f"{rep.recall:.4f})")
        print(f"     y pierde en nDCG ({fac.ndcg:.4f} frente a {rep.ndcg:.4f}).")
        print("\n     Traducido: encuentra MAS items relevantes y los ordena PEOR.")
        print("     Con una sola cifra, los dos equipos pueden publicar que ganaron,")
        print("     y los dos tienen razon. Esa es la pregunta del taller: cual de")
        print("     los dos elegirian ustedes, y para que producto.")
    print()
    return res


# ── B3 ──────────────────────────────────────────────────────────────────────
def b3(p) -> None:
    print("\n  B3 · La k tambien decide\n")
    print("     Cuantos items caben en la pantalla no es un detalle de diseno:")
    print("     es parte de la definicion de la metrica.\n")

    fac = factorizacion_implicita(p, factores=32, pasos=10)
    rep = repetir_lo_propio(p)
    print(f"       {'k':>4s} {'factorización':>16s} {'repetir lo propio':>19s} {'gana':>18s}")
    for k in (1, 3, 5, 10, 20):
        a = evaluar(p, fac, k=k)
        b = evaluar(p, rep, k=k)
        gana = "factorización" if a.recall > b.recall else "repetir lo propio"
        print(f"       {k:>4d} {a.recall:>16.4f} {b.recall:>19.4f} {gana:>18s}")

    print("\n     Lean la columna de la derecha de arriba abajo. El ganador no es")
    print("     una propiedad del modelo: es una propiedad del par (modelo, k).")
    print("\n     Regla que se lleva el curso: se reportan las cuatro cifras, con la")
    print("     k dicha en voz alta, o no se ha reportado nada.\n")


def main() -> None:
    print("\n  UPTC · Sesion 6 · Taller B · Las metricas")
    print("  Lo dirigen los grupos 3 y 4")
    b1()
    print("  Cargando ASSISTments...", flush=True)
    d = cargar()
    p = particion_temporal(d)
    print(f"  Particion temporal, {p.usuarios_evaluables:,} usuarios evaluables.")
    b2(p)
    b3(p)


if __name__ == "__main__":
    main()
