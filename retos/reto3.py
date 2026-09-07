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
):
    """
    Moldeado de recompensa basado en potenciales (Potential-Based Reward Shaping)
    conforme a la teoría de Ng, Harada & Russell (1999).
    
    Conceptualmente mapea la reducción del vacío de conocimiento (distancia a la meta)
    evitando que el agente explote el sistema quedándose en bucles locales.
    """
    # 1. Obtenemos las distancias al objetivo (nuestro vacío de conocimiento)
    dist_anterior = pasos_hasta_la_meta(anterior)
    dist_siguiente = pasos_hasta_la_meta(siguiente)
    
    # 2. Definimos las funciones de potencial de estado Phi(s)
    # A menor distancia, mayor potencial (menos negativo).
    phi_anterior = -float(dist_anterior)
    phi_siguiente = -float(dist_siguiente)
    
    # 3. Factor de descuento del GridWorld
    gamma = 0.9
    
    # 4. Formulamos el shaping: F = gamma * Phi(s') - Phi(s)
    # Si es un estado terminal (llegó a la meta), el potencial futuro es 0 por definición.
    if terminal:
        shaping = 0.0 - phi_anterior
    else:
        shaping = (gamma * phi_siguiente) - phi_anterior
        
    return shaping
