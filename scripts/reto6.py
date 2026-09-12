"""Arnes del reto 6: comprueba que su recomendador gana sin encoger el catalogo.

    uv run python scripts/reto6.py

Mide con particion TEMPORAL, que es la unica que responde a la pregunta que
importa, y ensena siempre las cuatro cifras.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from retos.reto6 import mi_recomendador  # noqa: E402

from rlrs.recomendacion import (  # noqa: E402
    cargar,
    evaluar,
    particion_aleatoria,
    particion_temporal,
    por_popularidad,
    repetir_lo_propio,
)

K = 10
MINIMO_RECALL = 0.7100
MINIMO_COBERTURA = 0.50


def main() -> int:
    print("\n  Reto 6 · Gane a dos lineas de codigo\n")
    print("  Cargando ASSISTments...", flush=True)
    datos = cargar()
    p = particion_temporal(datos)
    print(f"  Particion temporal · {p.usuarios_evaluables:,} usuarios · {p.n_items} items\n")

    mio = evaluar(p, mi_recomendador(p), k=K, nombre="SU RECOMENDADOR")
    rep = evaluar(p, repetir_lo_propio(p), k=K)
    pop = evaluar(p, por_popularidad(p), k=K)

    print(f"  {'':<20s} {'Recall@10':>10s} {'nDCG@10':>9s} {'cobertura':>11s} {'novedad':>9s}")
    for r in (mio, rep, pop):
        print(f"  {r.nombre:<20s} {r.recall:>10.4f} {r.ndcg:>9.4f} "
              f"{r.cobertura:>11.3f} {r.novedad:>9.2f}")

    ok_r = mio.recall >= MINIMO_RECALL
    ok_c = mio.cobertura >= MINIMO_COBERTURA
    print(f"\n  Criterios\n")
    print(f"    Recall@{K}     {mio.recall:>8.4f}   hace falta >= {MINIMO_RECALL:.4f}   "
          f"{'sí' if ok_r else 'NO'}")
    print(f"    Cobertura     {mio.cobertura:>8.3f}   hace falta >= {MINIMO_COBERTURA:.2f}     "
          f"{'sí' if ok_c else 'NO'}")

    if ok_r and ok_c:
        print("\n  SUPERADO.  Le gana a la linea base y no encogio el catalogo.")
    else:
        print("\n  NO SUPERADO")
        if not ok_r:
            print(f"    Le falta Recall. La linea base saca {rep.recall:.4f} repitiendo lo que")
            print("    el usuario ya hizo. Si su modelo no lo aprovecha, esta tirando la")
            print("    senal mas fuerte que hay en estos datos.")
        if not ok_c:
            print(f"    Su cobertura es {mio.cobertura:.3f}: le esta recomendando lo mismo a")
            print("    casi todo el mundo. Sube el Recall y encoge el catalogo.")

    # Lo que hay que anotar, se haya superado o no.
    print("\n  ── Para la bitacora ──")
    if mio.recall > rep.recall and mio.ndcg < rep.ndcg:
        print(f"  Su recomendador GANA en Recall ({mio.recall:.4f} frente a {rep.recall:.4f})")
        print(f"  y PIERDE en nDCG ({mio.ndcg:.4f} frente a {rep.ndcg:.4f}). Encuentra mas")
        print("  items relevantes y los ordena peor. Con una sola cifra podria declararse")
        print("  ganador, y con la otra tambien podria hacerlo la linea base. Reporte las dos.")
    elif mio.recall <= rep.recall:
        print(f"  Su recomendador ({mio.recall:.4f}) no le gana a repetir lo que el usuario")
        print(f"  ya hizo ({rep.recall:.4f}). Antes de tocar nada, escriba por que dos lineas")
        print("  de codigo le ganan a lo que usted escribio. La respuesta esta en los datos,")
        print("  no en su modelo.")
    else:
        print(f"  Su recomendador gana en las dos cifras. Es poco comun: escriba que decision")
        print("  concreta cree que lo consiguio, porque eso es lo que se pregunta despues.")

    # Y la comparacion que nadie pide y todo el mundo deberia hacer.
    pa = particion_aleatoria(datos)
    mio_a = evaluar(pa, mi_recomendador(pa), k=K, nombre="el suyo, barajando")
    if mio_a.ndcg > mio.ndcg:
        subida = 100 * (mio_a.ndcg - mio.ndcg) / max(mio.ndcg, 1e-9)
        print(f"\n  Y una ultima, por curiosidad: su MISMO recomendador, medido con particion")
        print(f"  aleatoria en vez de temporal, saca nDCG {mio_a.ndcg:.4f} en vez de {mio.ndcg:.4f}.")
        print(f"  Un {subida:.0f} % mas alto sin haber cambiado una linea. Eso es lo que se")
        print("  gana barajando, y es lo que hay en muchos articulos publicados.")
    print()
    return 0 if (ok_r and ok_c) else 1


if __name__ == "__main__":
    sys.exit(main())
