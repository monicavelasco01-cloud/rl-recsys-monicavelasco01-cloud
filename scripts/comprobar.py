#!/usr/bin/env python
"""Comprueba que tu copia del repositorio esta completa y al dia.

    uv run python scripts/comprobar.py

Este guion existe por un fallo real: la plantilla crecio despues de que
ustedes clonaran, y varios archivos del laboratorio no estaban en sus copias.
El sintoma era feo, un "No such file or directory" en mitad de un ejercicio,
y la causa no se veia por ningun lado.

Comprueba dos cosas distintas:

  1. Que los archivos ESTEN.
  2. Que esten AL DIA. Un archivo puede existir y ser de una version anterior,
     que es peor que si faltara: no falla al abrirlo, falla al usarlo. Por eso
     ademas de la lista de archivos hay una lista de piezas que tienen que
     poder importarse.

No modifica nada. Se puede ejecutar cuantas veces haga falta.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Archivos que el modulo necesita, con la sesion en la que se usan.
ARCHIVOS = [
    ("src/rlrs/envs.py", "el entorno"),
    ("src/rlrs/dp.py", "sesión 1"),
    ("src/rlrs/evaluation.py", "el arnés de medición"),
    ("src/rlrs/policies.py", "las políticas"),
    ("src/rlrs/td.py", "sesión 2"),
    ("scripts/verificar.py", "puesta a punto"),
    ("scripts/variante.py", "laboratorio 1"),
    ("experiments/run.py", "sesión 1"),
    ("experiments/divergencia.py", "laboratorio 1, ejercicio 6"),
    ("experiments/sin_modelo.py", "sesión 2 y laboratorio 2"),
    ("notebooks/00-bienvenida.ipynb", "antes de la sesión 1"),
    ("notebooks/01-laboratorio.ipynb", "bitácora del laboratorio 1"),
    ("notebooks/02-laboratorio.ipynb", "bitácora del laboratorio 2"),
    ("tests/test_dp.py", "pruebas"),
    ("tests/test_envs.py", "pruebas"),
    ("tests/test_evaluation.py", "pruebas"),
    ("tests/test_td.py", "pruebas"),
    ("configs/base.yaml", "sesión 1"),
]

# Piezas que tienen que poder importarse. Detectan el caso peor: el archivo
# esta, pero es de una version anterior y le falta la mitad.
PIEZAS = [
    ("rlrs.envs", "GridWorld", "el entorno"),
    ("rlrs.envs", "acantilado", "el acantilado de la sesión 2"),
    ("rlrs.dp", "value_iteration", "iteración de valor"),
    ("rlrs.dp", "q_value", "el respaldo de Bellman"),
    ("rlrs.evaluation", "evaluate", "el arnés de medición"),
    ("rlrs.policies", "GreedyTabularPolicy", "la política ávida"),
    ("rlrs.policies", "EpsilonAvidaPolicy", "la política que explora"),
    ("rlrs.td", "mc_control", "Monte Carlo"),
    ("rlrs.td", "sarsa", "SARSA"),
    ("rlrs.td", "q_learning", "Q-learning"),
    ("rlrs.td", "error_frente_a", "la medida del error"),
]

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"
if sys.platform == "win32":
    try:  # en PowerShell moderno funciona; en la consola vieja, no
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        VERDE = ROJO = AMARILLO = GRIS = FIN = ""


def main() -> int:
    print()
    print("  Comprobación de la copia del repositorio")
    print(f"  {GRIS}{RAIZ}{FIN}")
    print()

    faltan = [(ruta, para) for ruta, para in ARCHIVOS if not (RAIZ / ruta).exists()]
    if faltan:
        print(f"  {ROJO}Faltan {len(faltan)} de {len(ARCHIVOS)} archivos:{FIN}")
        for ruta, para in faltan:
            print(f"    {ROJO}·{FIN} {ruta:<38} {GRIS}({para}){FIN}")
    else:
        print(f"  {VERDE}[ OK ]{FIN}  Están los {len(ARCHIVOS)} archivos.")

    sys.path.insert(0, str(RAIZ / "src"))
    viejas = []
    for modulo, pieza, para in PIEZAS:
        try:
            mod = importlib.import_module(modulo)
            if not hasattr(mod, pieza):
                viejas.append((f"{modulo}.{pieza}", para, "el archivo está pero es de una versión anterior"))
        except ImportError as e:
            viejas.append((f"{modulo}.{pieza}", para, str(e)))

    if viejas:
        print()
        print(f"  {AMARILLO}Faltan {len(viejas)} de {len(PIEZAS)} piezas del código:{FIN}")
        for nombre, para, motivo in viejas:
            print(f"    {AMARILLO}·{FIN} {nombre:<38} {GRIS}({para}){FIN}")
            print(f"      {GRIS}{motivo}{FIN}")
    elif not faltan:
        print(f"  {VERDE}[ OK ]{FIN}  Las {len(PIEZAS)} piezas del código se importan bien.")

    print()
    print("  " + "-" * 68)
    if not faltan and not viejas:
        print(f"  {VERDE}Tu copia está completa y al día. Puedes hacer el laboratorio.{FIN}")
        print()
        return 0

    print(f"  {ROJO}Tu copia está incompleta.{FIN}  Para arreglarlo:")
    print()
    print("    1. Descomprime plantilla-actualizada.zip")
    print("    2. Copia el CONTENIDO de la carpeta plantilla-repo encima de esta")
    print("       carpeta, aceptando reemplazar lo que pregunte")
    print("    3. Vuelve a ejecutar:  uv run python scripts/comprobar.py")
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
