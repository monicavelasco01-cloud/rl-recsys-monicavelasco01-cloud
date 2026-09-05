"""Reto de la sesion 4 · Un agente que se rinde.

═══════════════════════════════════════════════════════════════════════════
QUE PASA
═══════════════════════════════════════════════════════════════════════════

El actor-critico de ``rlrs.pg`` colapsa en la mitad de las semillas. No
aprende despacio: **deja de aprender**. Lo viste en la parte 4 del
experimento:

    tramo             retorno medio   desviacion   llegan a la meta
    0 a 100                 -3.5056       0.9940                25 %
    200 a 300               -3.9776       0.2229                 1 %
    500 a 600               -4.0000       0.0000                 0 %

La columna que importa es la de la desviacion. Cuando llega a cero, todos los
episodios dan lo mismo, y un metodo que aprende comparando episodios se queda
sin nada que comparar.

═══════════════════════════════════════════════════════════════════════════
QUE HAY QUE HACER
═══════════════════════════════════════════════════════════════════════════

Dos cosas, y la primera se entrega aunque la segunda no salga.

1. **El diagnostico, escrito en tu bitacora antes de tocar el codigo.** Por
   que crees que pasa. Que cantidad se hace cero primero. Por que no se
   arregla con mas episodios.

2. **El arreglo.** Rellena ``mi_agente`` para que no colapse en ninguna
   semilla. Tienes todas las piezas de ``rlrs.pg`` disponibles y puedes mover
   lo que quieras: el metodo, la tasa de aprendizaje, la linea base, el
   tamano del lote, el termino de entropia.

    uv run python scripts/reto4.py        lo evalua con semillas que no ves
    uv run pytest tests/test_reto4.py     comprueba el contrato

═══════════════════════════════════════════════════════════════════════════

Una pista que no es una respuesta: el problema no esta en cuanto aprende, sino
en que deja de haber informacion que aprender. Cualquier arreglo tiene que
atacar eso.
"""

from __future__ import annotations

from rlrs.pg import EntrenamientoPG, actor_critico, reinforce  # noqa: F401


def mi_agente(env, phi, episodes: int, gamma: float, seed: int) -> EntrenamientoPG:
    """Entrena un agente de gradiente de politica que no colapse.

    Tiene que devolver lo que devuelven ``actor_critico`` o ``reinforce``, es
    decir un ``EntrenamientoPG``. Lo que hagas por dentro es cosa tuya.

    Parameters
    ----------
    env, phi:
        El entorno y la funcion de caracteristicas. Se los pasa el arnes.
    episodes, gamma, seed:
        Hay que respetarlos. Entrenar mas episodios de los que te dan, o con
        otra semilla, no cuenta como arreglo.
    """
    # ── tu respuesta va aqui ──────────────────────────────────────────────
    return actor_critico(env, phi, episodes=episodes, gamma=gamma, seed=seed)
