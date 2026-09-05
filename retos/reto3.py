"""Reto de la sesion 3 · Escriba usted la recompensa.

═══════════════════════════════════════════════════════════════════════════
QUE HAY QUE HACER
═══════════════════════════════════════════════════════════════════════════

El agente de la cuadricula tarda mucho en encontrar la meta porque durante
cientos de episodios no recibe ninguna senal util: solo el coste de cada paso.
Su trabajo es **darle una pista**, escribiendo una recompensa extra.

Rellene ``mi_moldeado``. Recibe tres cosas y devuelve un numero, que se suma a
la recompensa que el entorno ya entrega.

    anterior   la casilla donde estaba, como (fila, columna). Puede ser None
               en el primer paso.
    siguiente  la casilla a la que acaba de llegar.
    terminal   True si ``siguiente`` termina el episodio.

No hay ninguna restriccion sobre lo que puede escribir. Puede usar la
distancia a la meta, la fila, la columna, lo que se le ocurra.

═══════════════════════════════════════════════════════════════════════════
COMO SE SABE SI FUNCIONO
═══════════════════════════════════════════════════════════════════════════

    uv run python scripts/reto3.py        entrena y mide, e imprime el veredicto
    uv run pytest tests/test_reto3.py     comprueba el contrato

**Antes de ejecutar nada, escriba su prediccion en la bitacora.** Que espera
que haga su agente. Cuantos pasos va a tardar. Que retorno va a sacar.

Aviso, y va en serio: es muy probable que su primera version obtenga un
retorno estupendo y sea un desastre. Cuando eso pase, no lo arregle todavia.
Anotelo, que de eso trata la clase.
"""

from __future__ import annotations

# Estas dos las puede mover libremente.
GAMMA = 0.9      # tiene que ser el mismo descuento con el que se entrena
ESCALA = 0.5     # cuanto pesa su pista frente al coste del paso, que es -0,04

META = (0, 11)   # la esquina de arriba a la derecha de la sala de 8 x 12


def pasos_hasta_la_meta(pos: tuple[int, int]) -> int:
    """Cuantos pasos faltan hasta la meta, contando por la rejilla.

    Se la dejo hecha para que no pierda tiempo en esto. Usela o no la use.
    """
    return abs(pos[0] - META[0]) + abs(pos[1] - META[1])


def mi_moldeado(
    anterior: tuple[int, int] | None,
    siguiente: tuple[int, int],
    terminal: bool,
) -> float:
    """La recompensa extra de una transicion. **Esto es lo que usted escribe.**"""
    # ── su respuesta va aqui ──────────────────────────────────────────────
    return 0.0
