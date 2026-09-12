"""Arnes del reto 7: comprueba que su recomendador sobrevive a un usuario nuevo.

    uv run python scripts/reto7.py

Mide dos veces. Primero como siempre, con particion temporal. Despues con
usuarios apartados del ajuste, que es la condicion que separa a las
arquitecturas de las que solo aprenden una tabla.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from retos.reto7 import mi_recomendador  # noqa: E402

from rlrs.recomendacion import (  # noqa: E402
    cargar,
    evaluar,
    dos_torres,
    particion_temporal,
    repetir_lo_propio,
    usuarios_nuevos,
)

K = 10
MINIMO_NORMAL = 0.7100
MINIMO_NUEVOS = 0.6500
FRACCION_NUEVOS = 0.2


def main() -> int:
    print("\n  Reto 7 · El usuario que llega mañana\n")
    print("  Cargando ASSISTments...", flush=True)
    datos = cargar()
    normal = particion_temporal(datos)
    frio = usuarios_nuevos(normal, fraccion=FRACCION_NUEVOS, seed=0)
    print(f"  Ajuste con {len(frio.ajuste):,} estudiantes · "
          f"{len(frio.train):,} usuarios nuevos apartados\n")

    print("  Midiendo, unos segundos...", flush=True)
    filas = []
    for etiqueta, constructor in (("SU RECOMENDADOR", mi_recomendador),
                                  ("repetir lo propio", repetir_lo_propio),
                                  ("dos torres", dos_torres)):
        a = evaluar(normal, constructor(normal), k=K)
        b = evaluar(frio, constructor(frio), k=K)
        filas.append((etiqueta, a, b))

    print(f"\n  {'':<20s}{'partición normal':>22s}{'usuarios nuevos':>30s}")
    print(f"  {'':<20s}{'Recall@' + str(K):>11s}{'nDCG':>11s}"
          f"{'Recall@' + str(K):>13s}{'nDCG':>10s}{'sin respuesta':>14s}")
    for etiqueta, a, b in filas:
        print(f"  {etiqueta:<20s}{a.recall:>11.4f}{a.ndcg:>11.4f}"
              f"{b.recall:>13.4f}{b.ndcg:>10.4f}{100 * b.sin_respuesta:>13.1f} %")

    _, mio_normal, mio_frio = filas[0]
    ok_n = mio_normal.recall >= MINIMO_NORMAL
    ok_f = mio_frio.recall >= MINIMO_NUEVOS

    print("\n  Criterios\n")
    print(f"    Partición normal   {mio_normal.recall:>8.4f}   "
          f"hace falta >= {MINIMO_NORMAL:.4f}   {'sí' if ok_n else 'NO'}")
    print(f"    Usuarios nuevos    {mio_frio.recall:>8.4f}   "
          f"hace falta >= {MINIMO_NUEVOS:.4f}   {'sí' if ok_f else 'NO'}")

    if ok_n and ok_f:
        print("\n  SUPERADO")
        print("    Su recomendador aguanta las dos condiciones. Escriba en la bitácora")
        print("    qué parte de su solución responde a cada una: casi nunca es la misma.")
    else:
        print("\n  NO SUPERADO")
        if not ok_f and mio_frio.sin_respuesta > 0.5:
            print(f"    A un {100 * mio_frio.sin_respuesta:.0f} % de los usuarios nuevos su")
            print("    recomendador no les devuelve nada. Eso no se arregla entrenando más:")
            print("    es que el modelo aprende un vector por usuario y de un usuario que")
            print("    no estaba no hay vector. Hace falta calcular el vector a partir del")
            print("    historial, o apoyarse en algo que no dependa de conocer al usuario.")
        elif not ok_f:
            print("    Aguanta la partición normal y se queda corto con los usuarios nuevos.")
        elif not ok_n:
            print("    Aguanta a los usuarios nuevos y se queda corto en la partición normal.")
            print("    Repetir lo propio hace exactamente eso: 0,7009 y 0,7056. Le falta")
            print("    la parte que aprende.")

    print("\n  ── Para la bitácora ──")
    print(f"  Su recomendador pierde {100 * (mio_normal.recall - mio_frio.recall) / max(mio_normal.recall, 1e-9):.1f} %"
          " de su Recall al pasar de usuarios")
    print("  conocidos a usuarios nuevos. Anote ese porcentaje y explíquelo. Un sistema")
    print("  en producción ve usuarios nuevos todos los días, y esa caída es la que")
    print("  decide si el sistema sirve el primer día o solo a partir del tercer mes.\n")
    return 0 if (ok_n and ok_f) else 1


if __name__ == "__main__":
    raise SystemExit(main())
