#!/usr/bin/env python
"""Tu variante del laboratorio.

    uv run python scripts/variante.py

Cada estudiante trabaja sobre una configuracion ligeramente distinta del mismo
entorno. No es para dificultarlo: es para que en la puesta en comun haya nueve
resultados que comparar en vez de uno repetido diecisiete veces, y para que
copiar un numero no sirva de nada.

La variante se deduce de tu usuario de GitHub, que sale de `git config`, asi
que es siempre la misma para ti y no hay que repartir nada. El algoritmo esta
a la vista y no tiene sorpresas: se resume el usuario, se toma el resto de la
division, y eso indexa la tabla.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

# Nueve variantes. Solo se mueven el ruido y el coste por paso; gamma se queda
# en 0.9 en todas para que los resultados sigan siendo comparables entre si.
RUIDOS = (0.0, 0.2, 0.4)
COSTES = (-0.02, -0.04, -0.10)


def usuario_de_git() -> str | None:
    """Saca el usuario del remoto de GitHub, o del nombre configurado en Git."""
    for orden in (
        ["git", "config", "--get", "remote.origin.url"],
        ["git", "config", "--get", "user.name"],
    ):
        try:
            salida = subprocess.run(orden, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        texto = salida.stdout.strip()
        if not texto:
            continue
        if "github.com" in texto:
            # https://github.com/USUARIO/repo.git  o  git@github.com:USUARIO/repo.git
            resto = texto.split("github.com")[1].lstrip(":/")
            if "/" in resto:
                return resto.split("/")[0]
        else:
            return texto
    return None


def variante_de(usuario: str) -> tuple[float, float]:
    """Devuelve ``(ruido, coste_por_paso)`` para ese usuario."""
    digesto = hashlib.sha256(usuario.strip().lower().encode("utf-8")).digest()
    n = int.from_bytes(digesto[:4], "big")
    return RUIDOS[n % 3], COSTES[(n // 3) % 3]


def main() -> None:
    usuario = sys.argv[1] if len(sys.argv) > 1 else usuario_de_git()
    if not usuario:
        print(
            "\n  No pude deducir tu usuario.\n"
            "  Pasalo a mano:  uv run python scripts/variante.py TU-USUARIO\n"
        )
        raise SystemExit(1)

    ruido, coste = variante_de(usuario)

    print(f"\n  Variante de  {usuario}\n")
    print(f"    ruido           {ruido}")
    print(f"    coste por paso  {coste}")
    print(f"    gamma           0.9   (igual para todos)")
    print()
    print("  --------------------------------------------------------------")
    print("  ESTO NO SE ESCRIBE EN LA TERMINAL.")
    print("  --------------------------------------------------------------")
    print()
    print("  Son dos numeros que van dentro de tu cuaderno de bitacora:")
    print()
    print("    1. Abre en VS Code el archivo  notebooks/01-laboratorio.ipynb")
    print("       (en el laboratorio 2 sera  notebooks/02-laboratorio.ipynb)")
    print()
    print("    2. Busca la celda que empieza por RUIDO = None")
    print()
    print("    3. Cambia los dos None por estos numeros, y ejecuta la celda:")
    print()
    print(f"           RUIDO = {ruido}")
    print(f"           COSTE = {coste}")
    print()
    print("  La linea  mi_env = GridWorld(noise=RUIDO, step_reward=COSTE)  ya")
    print("  esta escrita debajo en el cuaderno: no hay que tocarla. Con esos")
    print("  dos numeros puestos, construye tu entorno sola.")
    print()
    print("  Anota tambien los dos numeros en la primera pagina de la bitacora:")
    print("  todo lo que midas durante el laboratorio es sobre ESE entorno.")
    print()


if __name__ == "__main__":
    main()
