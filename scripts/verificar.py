#!/usr/bin/env python
"""Prueba de paso del Bloque 0 - N1.

Comprueba, una por una, las ocho condiciones que hacen falta para empezar el
modulo, y escribe ``entorno_verificado.json`` con el resultado.

    uv run python scripts/verificar.py

No arregla nada: solo dice que falta y por que. Cada fallo trae la instruccion
concreta para resolverlo, asi que se puede volver a ejecutar tantas veces como
haga falta hasta ver las ocho marcas en verde.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ANCHO = 72

VERDE, ROJO, AMAR, GRIS, NEGRITA, FIN = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)


class Verificacion:
    def __init__(self) -> None:
        self.resultados: list[dict] = []

    def comprobar(self, numero: int, titulo: str, funcion) -> bool:
        """Ejecuta una comprobacion y la reporta. ``funcion`` devuelve (ok, detalle, ayuda)."""
        try:
            ok, detalle, ayuda = funcion()
        except Exception as exc:  # noqa: BLE001 - queremos reportar cualquier fallo
            ok, detalle, ayuda = False, f"error inesperado: {exc}", "Revisa el mensaje anterior."
        marca = f"{VERDE}[ OK  ]{FIN}" if ok else f"{ROJO}[FALLA]{FIN}"
        print(f" {marca}  {numero}. {titulo}")
        print(f"         {GRIS}{detalle}{FIN}")
        if not ok:
            print(f"         {AMAR}-> {ayuda}{FIN}")
        self.resultados.append({"n": numero, "titulo": titulo, "ok": ok, "detalle": detalle})
        return ok


# --------------------------------------------------------------------- pruebas


def python_correcto():
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    detalle = f"Python {v.major}.{v.minor}.{v.micro} en {platform.system()} {platform.machine()}"
    return ok, detalle, "Necesitas Python 3.11 o superior. Ejecuta: uv python install 3.12"


def entorno_virtual():
    activo = sys.prefix != sys.base_prefix
    esperado = RAIZ / ".venv"
    aqui = Path(sys.prefix).resolve() == esperado.resolve()
    detalle = f"interprete en {sys.prefix}"
    if not activo:
        return False, detalle, "No estas dentro de un entorno virtual. Usa 'uv run python scripts/verificar.py'."
    if not aqui:
        return False, detalle, f"Estas en otro entorno. El del proyecto es {esperado}."
    return True, detalle + "  (es el .venv del proyecto)", ""


def dependencias():
    faltan, versiones = [], []
    for nombre, modulo in (("numpy", "numpy"), ("pyyaml", "yaml"), ("matplotlib", "matplotlib")):
        try:
            mod = __import__(modulo)
            versiones.append(f"{nombre} {getattr(mod, '__version__', '?')}")
        except ImportError:
            faltan.append(nombre)
    if faltan:
        return False, f"faltan: {', '.join(faltan)}", "Ejecuta: uv sync"
    return True, " · ".join(versiones), ""


def paquete_importable():
    try:
        import rlrs
    except ImportError as exc:
        return False, str(exc), "El paquete no esta instalado. Ejecuta: uv sync"
    ruta = Path(rlrs.__file__).parent
    return True, f"rlrs {rlrs.__version__} desde {ruta.relative_to(RAIZ) if RAIZ in ruta.parents else ruta}", ""


def el_proyecto_corre():
    from rlrs.dp import value_iteration
    from rlrs.envs import GridWorld
    from rlrs.evaluation import evaluate
    from rlrs.policies import GreedyTabularPolicy

    env = GridWorld()
    _, tabla, barridos = value_iteration(env, gamma=0.9)
    res = evaluate(env, GreedyTabularPolicy(tabla), episodes=100, base_seed=0)
    ok = res.success_rate > 0.7
    detalle = f"iteracion de valor converge en {barridos} barridos · tasa de exito {res.success_rate:.0%}"
    return ok, detalle, "El resultado no es el esperado. Vuelve a clonar el repositorio sin modificarlo."


def pruebas_pasan():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
    )
    salida = (proc.stdout or "") + (proc.stderr or "")
    lineas = [ln for ln in salida.strip().splitlines() if ln.strip()]
    resumen = lineas[-1] if lineas else "pytest no produjo salida"
    if "No module named pytest" in salida:
        return False, "pytest no esta instalado", "Ejecuta: uv sync"
    return proc.returncode == 0, resumen, "Ejecuta 'uv run pytest -q' y lee el primer error."


def git_configurado():
    def cfg(clave):
        r = subprocess.run(["git", "config", "--get", clave], capture_output=True, text=True)
        return r.stdout.strip()

    try:
        version = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return False, "git no encontrado", "Instala Git y vuelve a abrir la terminal."
    nombre, correo = cfg("user.name"), cfg("user.email")
    if not nombre or not correo:
        return False, f"{version} · falta user.name o user.email", (
            'Ejecuta: git config --global user.name "Tu Nombre" '
            'y git config --global user.email "tu@correo.com"'
        )
    return True, f"{version} · {nombre} <{correo}>", ""


def repositorio_listo():
    r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=RAIZ, capture_output=True, text=True)
    if r.returncode != 0:
        return False, "esta carpeta no es un repositorio git", "Crea tu repositorio desde la plantilla y clonalo; no descargues el ZIP."
    rama = subprocess.run(["git", "branch", "--show-current"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    remoto = subprocess.run(["git", "remote", "-v"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    if not remoto:
        return False, f"rama '{rama}' · sin remoto configurado", "Falta el remoto. Revisa 'Crear tu repositorio del curso' en el instructivo."
    origen = remoto.splitlines()[0].split()[1]
    return True, f"rama '{rama}' · {n} commits · remoto {origen}", ""


# ----------------------------------------------------------------------- salida


def main() -> int:
    print()
    print(f"{NEGRITA}  UPTC · Especializacion en Inteligencia Artificial{FIN}")
    print("  Aprendizaje por Refuerzo y Sistemas de Recomendacion")
    print(f"  {GRIS}Bloque 0 · N1 · Prueba de paso del entorno{FIN}")
    print("  " + "-" * ANCHO)
    print()

    v = Verificacion()
    pruebas = [
        ("Python instalado y con version suficiente", python_correcto),
        ("Entorno virtual del proyecto activo", entorno_virtual),
        ("Dependencias instaladas", dependencias),
        ("El paquete 'rlrs' se importa", paquete_importable),
        ("El proyecto se ejecuta y da el resultado esperado", el_proyecto_corre),
        ("Las pruebas automaticas pasan", pruebas_pasan),
        ("Git instalado y configurado", git_configurado),
        ("El repositorio esta clonado y con remoto", repositorio_listo),
    ]
    for i, (titulo, fn) in enumerate(pruebas, start=1):
        v.comprobar(i, titulo, fn)
        print()

    superadas = sum(1 for r in v.resultados if r["ok"])
    total = len(v.resultados)
    todo_ok = superadas == total

    huella = hashlib.sha256(
        "|".join(f"{r['n']}:{int(r['ok'])}" for r in v.resultados).encode()
    ).hexdigest()[:6].upper()
    codigo = f"UPTC-RL-{'OK' if todo_ok else 'NO'}-{superadas}{total}-{huella}"

    informe = {
        "modulo": "Aprendizaje por Refuerzo y Sistemas de Recomendacion",
        "prueba": "Bloque 0 / N1",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sistema": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "superadas": superadas,
        "total": total,
        "codigo": codigo,
        "detalle": v.resultados,
    }
    (RAIZ / "entorno_verificado.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("  " + "-" * ANCHO)
    if todo_ok:
        print(f"  {VERDE}{NEGRITA}{superadas} de {total} comprobaciones superadas. Tu entorno esta listo.{FIN}")
        print()
        print(f"  Codigo de verificacion:  {NEGRITA}{codigo}{FIN}")
        print()
        print("  Ultimo paso: sube el archivo generado a tu repositorio.")
        print(f"    {GRIS}git add entorno_verificado.json{FIN}")
        print(f'    {GRIS}git commit -m "Entorno verificado"{FIN}')
        print(f"    {GRIS}git push{FIN}")
    else:
        print(f"  {ROJO}{NEGRITA}{superadas} de {total} comprobaciones superadas.{FIN}")
        print("  Resuelve los puntos marcados en rojo y vuelve a ejecutar este guion.")
        print(f"  {GRIS}No pasa nada por ejecutarlo muchas veces: no modifica nada.{FIN}")
    print()
    print(f"  {GRIS}Informe escrito en entorno_verificado.json{FIN}")
    print()
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
