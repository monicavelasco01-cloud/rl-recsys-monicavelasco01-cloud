"""Ejecuta un experimento descrito por un archivo de configuracion.

    uv run python experiments/run.py --config configs/base.yaml

Este guion no contiene ningun algoritmo: los importa del paquete. Su unica
responsabilidad es leer la configuracion, orquestar y reportar. Esa separacion
es la que permite que el mismo algoritmo se ejecute desde aqui, desde una
prueba o desde un notebook, y de siempre lo mismo.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
from rlrs.evaluation import compare
from rlrs.policies import GreedyTabularPolicy, RandomPolicy

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento base del curso")
    parser.add_argument("--config", default="configs/base.yaml", help="ruta al archivo de configuracion")
    parser.add_argument("--quiet", action="store_true", help="no imprimir la rejilla")
    args = parser.parse_args()

    cfg = yaml.safe_load((RAIZ / args.config).read_text(encoding="utf-8"))

    env = GridWorld(**cfg["entorno"])
    gamma = cfg["agente"]["gamma"]

    values, table, sweeps = value_iteration(env, gamma=gamma)

    resultados = compare(
        env,
        [
            GreedyTabularPolicy(table, name="optima"),
            RandomPolicy(env.n_actions, seed=0),
        ],
        episodes=cfg["evaluacion"]["episodios"],
        base_seed=cfg["evaluacion"]["semilla_base"],
    )

    print(f"\nExperimento '{cfg['nombre']}'  ·  gamma = {gamma}  ·  {sweeps} barridos hasta converger\n")
    if not args.quiet:
        print(env.render_values(values, table))
        print()
    for r in resultados:
        print("  " + str(r))
    print()

    salida = RAIZ / "runs" / cfg["nombre"]
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resultado.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "sweeps": sweeps,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "resultados": [
                    {
                        "politica": r.policy_name,
                        "retorno_medio": round(r.mean, 4),
                        "ic95": [round(r.ci95[0], 4), round(r.ci95[1], 4)],
                        "tasa_exito": round(r.success_rate, 4),
                    }
                    for r in resultados
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Resultados guardados en runs/{cfg['nombre']}/resultado.json\n")


if __name__ == "__main__":
    main()
