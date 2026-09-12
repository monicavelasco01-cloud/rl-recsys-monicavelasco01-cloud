"""Decidir sin desplegar, medido.

    uv run python experiments/fuera_de_politica.py             las tres partes
    uv run python experiments/fuera_de_politica.py --parte 3   solo una

Tres partes:

  1. Los cuatro estimadores, en tres escenarios cada vez mas lejanos.
  2. **Veinte semillas**, porque una estimacion no es un estimador.
  3. **La condicion de soporte**, que es la unica que no se puede saltar.

Todos los numeros de la guia del sabado 19 salen de aqui.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlrs import fuera_de_politica as fp
from rlrs.recomendacion import cargar

N = 20000
USUARIOS = 800
SEMILLAS = 20


def _mundo():
    print("  Cargando ASSISTments...", flush=True)
    d = cargar()
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(d.secuencias))[:USUARIOS]
    pref = fp.preferencias([d.secuencias[u] for u in sel], d.n_items)
    pi0 = fp.politica_registro(pref, temperatura=1.0, epsilon=0.10)
    return pref, pi0


# ── 1 ───────────────────────────────────────────────────────────────────────
def parte1() -> None:
    print("\n  1 · Los cuatro estimadores, y hasta dónde llegan\n")
    pref, pi0 = _mundo()
    print(f"     Registro de {N:,} interacciones, generado por una política que")
    print(f"     no mira al usuario y explora un 10 %. Vale {fp.valor_verdadero(pref, pi0):.4f}.\n")

    escenarios = (
        ("casi la misma política", fp.politica_registro(pref, temperatura=1.1, epsilon=0.10)),
        ("personal, tibia", fp.politica_personal(pref, temperatura=0.30)),
        ("personal, decidida", fp.politica_personal(pref, temperatura=0.08)),
    )
    for nombre, pi1 in escenarios:
        ests, verdad, neff = fp.comparar(pref, pi0, pi1, n=N, seed=1)
        print(f"     ── {nombre}")
        print(f"        distancia a la de registro {fp.distancia(pi0, pi1):.3f} · "
              f"vale de verdad {verdad:.4f}")
        print(f"        tamaño efectivo {neff:,.0f} de {N:,}")
        for e in ests:
            print(f"        {e}")
        print()

    print("     Tres lecturas.\n")
    print("     **El directo falla siempre**, y falla con el intervalo más estrecho de")
    print("     los cuatro. Su modelo de recompensa ignora al usuario, así que no puede")
    print("     ver la ventaja de una política que sí lo mira. Está seguro y equivocado,")
    print("     que es la peor combinación posible en un informe.")
    print("\n     **Los tres que usan las propensiones aciertan** en los tres escenarios,")
    print("     y sus intervalos se ensanchan a medida que la política nueva se aleja.")
    print("     Ensancharse es lo correcto: la incertidumbre es real y la están diciendo.")
    print("\n     **El tamaño efectivo es el termómetro.** Cuando la política nueva se")
    print("     decide, veinte mil observaciones valen lo que unos cientos, y la anchura")
    print("     del intervalo lo refleja antes de que nadie se lleve una sorpresa.\n")


# ── 2 ───────────────────────────────────────────────────────────────────────
def parte2() -> None:
    print("\n  2 · Veinte semillas, porque una estimación no es un estimador\n")
    pref, pi0 = _mundo()
    pi1 = fp.politica_personal(pref, temperatura=0.08)
    verdad = fp.valor_verdadero(pref, pi1)
    print(f"     Política nueva decidida. Vale de verdad {verdad:.4f}.")
    print(f"     Se repite el registro entero {SEMILLAS} veces con semillas distintas.\n")

    acum: dict[str, list] = {}
    for s in range(SEMILLAS):
        ests, _, _ = fp.comparar(pref, pi0, pi1, n=N, seed=100 + s)
        for e in ests:
            acum.setdefault(e.nombre, []).append((e.valor, e.cubre))

    print(f"     {'estimador':<22}{'media':>10}{'desv.':>9}{'sesgo':>9}"
          f"{'RECM':>9}{'cubre':>9}")
    resumen = {}
    for nombre, vals in acum.items():
        v = np.array([x for x, _ in vals])
        cubre = float(np.mean([y for _, y in vals]))
        sesgo = float(v.mean() - verdad)
        recm = float(np.sqrt(np.mean((v - verdad) ** 2)))
        resumen[nombre] = (float(v.std(ddof=1)), sesgo, recm, cubre)
        print(f"     {nombre:<22}{v.mean():>10.4f}{v.std(ddof=1):>9.4f}"
              f"{sesgo:>9.4f}{recm:>9.4f}{100 * cubre:>8.0f} %")

    mejor = min(resumen, key=lambda n: resumen[n][2])
    print(f"\n     El de menor error cuadrático medio es **{mejor}**, "
          f"con {resumen[mejor][2]:.4f}.")
    if "SNIPS" in resumen and "IPS" in resumen:
        s_sd, i_sd = resumen["SNIPS"][0], resumen["IPS"][0]
        print(f"\n     SNIPS tiene {100 * (1 - s_sd / i_sd):.0f} % menos desviación que IPS "
              f"({s_sd:.4f} frente a {i_sd:.4f})")
        print("     a cambio de un sesgo pequeño. Ese cambio, sesgo por varianza, es la")
        print("     decisión de diseño de todo este tema, y casi siempre conviene.")
    directo = resumen.get("directo")
    if directo and directo[3] == 0.0:
        print(f"\n     Y el directo cubre la verdad en **0 de {SEMILLAS} semillas**. No es")
        print("     que tenga mala suerte: su intervalo mide la variabilidad del modelo,")
        print("     no la distancia del modelo a la realidad. Un intervalo estrecho")
        print("     alrededor del número equivocado sigue siendo el número equivocado.\n")


# ── 3 ───────────────────────────────────────────────────────────────────────
def parte3() -> None:
    print("\n  3 · La condición que no se puede saltar\n")
    pref, _ = _mundo()
    pi1 = fp.politica_personal(pref, temperatura=0.30)
    verdad = fp.valor_verdadero(pref, pi1)
    print("     Misma política nueva en los tres casos. Lo único que cambia es")
    print("     **cuántos ítems llegó a mostrar el sistema anterior**.")
    print(f"     La política nueva vale de verdad {verdad:.4f}.\n")

    casos = (("catálogo completo, 111 ítems", None),
             ("catálogo recortado a 20 ítems", 20),
             ("catálogo recortado a 10 ítems", 10))
    for nombre, catalogo in casos:
        pi0 = fp.politica_registro(pref, temperatura=1.0, epsilon=0.10, catalogo=catalogo)
        ests, _, neff = fp.comparar(pref, pi0, pi1, n=N, seed=1)
        masa = fp.masa_observable(pi0, pi1)
        print(f"     ── {nombre}")
        print(f"        masa de la política nueva que el registro pudo observar: "
              f"{100 * masa:.1f} %")
        print(f"        tamaño efectivo {neff:,.0f} de {N:,}")
        for e in ests:
            print(f"        {e}")
        print()

    print("     Esto es lo más incómodo del bloque.\n")
    print("     Con el catálogo recortado, **los cuatro estimadores fallan a la vez**, y")
    print("     cada uno en una dirección distinta: IPS se queda corto, SNIPS se pasa, y")
    print("     el directo y el doblemente robusto se disparan a cinco veces la verdad.")
    print("     Los cuatro con intervalos estrechos.")
    print("\n     Y ahora miren el tamaño efectivo: **sigue viéndose bien**. El diagnóstico")
    print("     en el que uno confía no detecta este fallo, porque los pesos que existen")
    print("     están perfectamente repartidos. Lo que falta no está en los pesos: falta")
    print("     el dato entero, y no hay dato que pesar.")
    print("\n     La comprobación que sí lo detecta es de una línea y hay que hacerla")
    print("     ANTES que nada: qué fracción de lo que haría la política nueva pudo")
    print("     llegar a observarse. Si no es 1, la cifra que salga no significa nada.\n")


PARTES = {1: parte1, 2: parte2, 3: parte3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parte", type=int, choices=sorted(PARTES), default=None)
    args = ap.parse_args()
    print("\n  UPTC · Sesión 8 · Decidir sin desplegar")
    for n in ([args.parte] if args.parte else sorted(PARTES)):
        PARTES[n]()


if __name__ == "__main__":
    main()
