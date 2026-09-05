"""Meridiano: la linea base del proyecto, y la trampa que hay que evitar.

    uv run python experiments/meridiano.py

Tres partes:

  1. Que tan bueno es el modelo de estudiante. Antes de construir nada encima,
     hay que saber cuanto vale el suelo.
  2. Las politicas triviales. El liston que hay que superar no es cero: es una
     rotacion por turnos, que no aprende nada y lo hace sorprendentemente bien.
  3. La trampa. Con la recompensa que sale sola, la politica optima es un
     desastre pedagogico con metricas magnificas.

La parte 3 es la sesion 3 otra vez, ahora sobre datos reales y sobre el
problema del proyecto. Si no la ves aqui, la vas a ver en tu informe.
"""

from __future__ import annotations

import numpy as np

from rlrs.meridiano import Meridiano, ajustar_pfa, calidad, medir

N_PASOS = 50
CATALOGO = 20
EPISODIOS = 200


def politicas(env: Meridiano, modelo, semilla: int = 0) -> dict:
    rng = np.random.default_rng(semilla)
    k = env.n_actions
    relleno = modelo.n_habilidades - k

    def p_actual(obs):
        return modelo.probabilidades(
            np.pad(obs[:k], (0, relleno)), np.pad(obs[k:], (0, relleno))
        )[:k]

    turno = [-1]

    def por_turnos(obs):
        turno[0] = (turno[0] + 1) % k
        return turno[0]

    return {
        "siempre la misma": lambda obs: 0,
        "al azar": lambda obs: int(rng.integers(k)),
        "por turnos": por_turnos,
        "la mas facil ahora": lambda obs: int(np.argmax(p_actual(obs))),
        "la mas dificil ahora": lambda obs: int(np.argmin(p_actual(obs))),
    }


def main() -> None:
    print("\n  Meridiano · linea base del proyecto\n")

    # ── 1 ─────────────────────────────────────────────────────────────────
    print("  1 · El modelo de estudiante\n")
    modelo = ajustar_pfa()
    c = calidad(modelo)
    print(f"     acierto sobre el conjunto de prueba   {c['acierto']:.4f}")
    print(f"     AUC                                   {c['auc']:.4f}")
    print(f"     tasa base (predecir siempre lo comun) {c['tasa_base']:.4f}")
    print(f"     interacciones de prueba               {c['interacciones']:,}")
    print("\n     Es mejor que la tasa base, y no por mucho. Eso es lo normal en")
    print("     trazado de conocimiento y conviene saberlo antes de creerse nada:")
    print("     el suelo sobre el que vas a construir tiene esta calidad.\n")

    # ── 2 ─────────────────────────────────────────────────────────────────
    print("  2 · Politicas triviales, con la recompensa correcta (dominio)\n")
    env = Meridiano(modelo, recompensa="dominio", n_pasos=N_PASOS, catalogo=CATALOGO)
    print(f"     {env.n_actions} habilidades, {env.n_caracteristicas} numeros de estado, "
          f"{N_PASOS} ejercicios por episodio\n")
    print(f"     {'politica':<22s} {'ganancia':>10s} {'aciertos':>10s} {'habs':>7s}")
    resultados = {}
    for nombre, pol in politicas(env, modelo).items():
        r = medir(env, pol, episodios=EPISODIOS)
        resultados[nombre] = r
        print(f"     {nombre:<22s} {r['ganancia']:>+10.4f} {r['aciertos']:>10.3f} "
              f"{r['habilidades_distintas']:>7.1f}")

    mejor = max(resultados.items(), key=lambda kv: kv[1]["ganancia"])
    print(f"\n     El liston es {mejor[0]}, con {mejor[1]['ganancia']:+.4f}. Una rotacion")
    print("     que no mira el estado, no aprende nada y no tiene parametros. Si tu")
    print("     agente no la supera, no has demostrado nada todavia.\n")

    # ── 3 ─────────────────────────────────────────────────────────────────
    print("  3 · La trampa: que pasa si la recompensa son los aciertos\n")
    facil = resultados["la mas facil ahora"]
    turnos = resultados["por turnos"]
    print(f"     {'':<22s} {'ganancia':>10s} {'aciertos':>10s} {'habs':>7s}")
    print(f"     {'por turnos':<22s} {turnos['ganancia']:>+10.4f} "
          f"{turnos['aciertos']:>10.3f} {turnos['habilidades_distintas']:>7.1f}")
    print(f"     {'la mas facil ahora':<22s} {facil['ganancia']:>+10.4f} "
          f"{facil['aciertos']:>10.3f} {facil['habilidades_distintas']:>7.1f}")

    print("\n     Lee la fila de abajo despacio. Esa politica responde bien el")
    print(f"     {facil['aciertos']:.1%} de los ejercicios y ensena {turnos['ganancia'] / max(facil['ganancia'], 1e-9):.0f} veces menos que")
    print(f"     una rotacion tonta. Toca {facil['habilidades_distintas']:.1f} habilidades distintas de {env.n_actions}.")
    print("\n     Y no esta rota: es EXACTAMENTE la politica optima si la recompensa")
    print("     son los aciertos. Un agente entrenado asi te va a ensenar un tablero")
    print("     con 98 % de acierto y va a estar poniendo la misma pregunta facil")
    print("     cincuenta veces seguidas.")
    print("\n     Esa es la primera decision de diseno de tu proyecto, y es la que")
    print("     mas consecuencias tiene. La sesion 3 iba de esto.\n")


if __name__ == "__main__":
    main()
