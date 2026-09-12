"""Torneo ciego: los recomendadores de los seis equipos, con un protocolo sorteado.

    uv run python scripts/torneo.py --sortear --seed 7
    uv run python scripts/torneo.py --particion temporal --k 10 --metrica recall

Espera una carpeta con un archivo por equipo:

    torneo/
      grupo1.py
      grupo2.py
      ...

Cada archivo tiene que exponer ``mi_recomendador(particion, k=50)``, que es
exactamente la firma de ``retos/reto7.py``. Lo normal es copiar el `reto7.py`
de cada equipo y renombrarlo.

Se puede cambiar la carpeta con ``--carpeta``.

El protocolo se sortea con ``--sortear``, y son tres cosas: la particion, la k
y la metrica que manda. Veinticuatro combinaciones. Ninguna se anuncia antes de
la sesion: si se sabe, cada equipo la optimiza y el torneo mide otra cosa.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from rlrs.recomendacion import (  # noqa: E402
    cargar,
    evaluar,
    particion_temporal,
    repetir_lo_propio,
    usuarios_nuevos,
)

PARTICIONES = ("temporal", "nuevos")
KS = (1, 5, 10, 20)
METRICAS = ("recall", "ndcg", "cobertura")


def sortear(seed: int) -> tuple[str, int, str]:
    rng = np.random.default_rng(seed)
    return (str(rng.choice(PARTICIONES)), int(rng.choice(KS)), str(rng.choice(METRICAS)))


def cargar_equipos(carpeta: Path) -> dict:
    """Importa un `mi_recomendador` por archivo. Un archivo que revienta no tumba el torneo."""
    equipos = {}
    for ruta in sorted(carpeta.glob("*.py")):
        if ruta.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"torneo_{ruta.stem}", ruta)
            modulo = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = modulo
            spec.loader.exec_module(modulo)
            equipos[ruta.stem] = modulo.mi_recomendador
        except Exception as e:                       # noqa: BLE001
            equipos[ruta.stem] = e
    return equipos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", type=Path, default=RAIZ / "torneo")
    ap.add_argument("--sortear", action="store_true", help="sortea el protocolo")
    ap.add_argument("--seed", type=int, default=0, help="semilla del sorteo")
    ap.add_argument("--particion", choices=PARTICIONES, default="temporal")
    ap.add_argument("--k", type=int, choices=KS, default=10)
    ap.add_argument("--metrica", choices=METRICAS, default="recall")
    args = ap.parse_args()

    if args.sortear:
        args.particion, args.k, args.metrica = sortear(args.seed)

    etiqueta = {"recall": f"Recall@{args.k}", "ndcg": f"nDCG@{args.k}",
                "cobertura": "cobertura"}[args.metrica]
    print("\n  Torneo ciego · protocolo sorteado" if args.sortear
          else "\n  Torneo ciego · protocolo fijado a mano")
    print(f"  partición {args.particion} · k = {args.k} · manda {etiqueta}\n")

    if not args.carpeta.exists():
        print(f"  No existe la carpeta {args.carpeta}.")
        print("  Cree una con un archivo por equipo: grupo1.py, grupo2.py, ...")
        print("  Cada uno es el reto7.py de ese equipo, renombrado.\n")
        return 2

    print("  Cargando ASSISTments...", flush=True)
    datos = cargar()
    base = particion_temporal(datos)
    p = base if args.particion == "temporal" else usuarios_nuevos(base, fraccion=0.2, seed=0)

    equipos = cargar_equipos(args.carpeta)
    if not equipos:
        print(f"  La carpeta {args.carpeta} está vacía.\n")
        return 2

    print(f"  {len(equipos)} equipos · midiendo, esto tarda...\n", flush=True)
    filas = []
    for nombre, constructor in equipos.items():
        if isinstance(constructor, Exception):
            filas.append((nombre, None, str(constructor)))
            continue
        try:
            filas.append((nombre, evaluar(p, constructor(p), k=args.k), None))
        except Exception:                            # noqa: BLE001
            filas.append((nombre, None, traceback.format_exc(limit=1).strip().splitlines()[-1]))

    clave = {"recall": lambda r: r.recall, "ndcg": lambda r: r.ndcg,
             "cobertura": lambda r: r.cobertura}[args.metrica]
    ok = sorted([f for f in filas if f[1] is not None], key=lambda f: -clave(f[1]))
    mal = [f for f in filas if f[1] is None]

    print(f"  {'puesto':>7}  {'equipo':<12}{'Recall@' + str(args.k):>12}"
          f"{'nDCG':>9}{'cobertura':>12}{'novedad':>10}{'sin resp.':>11}")
    for i, (nombre, r, _) in enumerate(ok, start=1):
        print(f"  {i:>7}  {nombre:<12}{r.recall:>12.4f}{r.ndcg:>9.4f}"
              f"{r.cobertura:>12.3f}{r.novedad:>10.2f}{100 * r.sin_respuesta:>10.1f} %")
    for nombre, _, error in mal:
        print(f"  {'—':>7}  {nombre:<12}{'no ejecuta':>12}   {error[:44]}")

    # La referencia que siempre se enseña, se haya pedido o no.
    ref = evaluar(p, repetir_lo_propio(p), k=args.k)
    print(f"\n  {'ref.':>7}  {'la línea base':<12}{ref.recall:>12.4f}{ref.ndcg:>9.4f}"
          f"{ref.cobertura:>12.3f}{ref.novedad:>10.2f}{100 * ref.sin_respuesta:>10.1f} %")

    mejores = [n for n, r, _ in ok if clave(r) > clave(ref)]
    print(f"\n  Le ganan a «repetir lo propio» en {etiqueta}: "
          f"{len(mejores)} de {len(ok)}.")
    if mal:
        print(f"  {len(mal)} no ejecutan. Un sistema que no se puede ejecutar no se")
        print("  puede evaluar, y eso también es un resultado.")
    print("\n  Este puesto vale para ESTE protocolo. Con otro, el orden puede ser otro.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
