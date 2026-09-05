"""Pruebas del contrato del reto 3.

Estas pruebas no miran si su agente saca buen retorno: eso lo mide
``scripts/reto3.py``. Lo que miran es **que propiedad tiene la funcion que
usted escribio**, sin entrenar nada, en menos de un segundo.

Cuando una falle, lea el mensaje entero. Cada mensaje dice que propiedad falta
y por que importa. Esa es la teoria de la sesion, escrita al reves: primero se
tropieza con ella y el sabado se le pone nombre.
"""

from __future__ import annotations

import pytest

from retos.reto3 import GAMMA, mi_moldeado

# Mientras el reto siga sin contestar, estas pruebas se saltan en vez de
# fallar. Asi ``uv run pytest`` sigue saliendo limpio y estas pruebas solo
# hablan cuando hay algo que decir.
_MUESTRAS = [((7, 0), (6, 0), False), ((3, 5), (3, 6), False), ((0, 10), (0, 11), True)]
_SIN_CONTESTAR = all(abs(float(mi_moldeado(a, b, t))) < 1e-12 for a, b, t in _MUESTRAS)
pytestmark = pytest.mark.skipif(
    _SIN_CONTESTAR,
    reason="El reto 3 todavia no esta contestado: rellene 'mi_moldeado' en retos/reto3.py",
)

# Un recorrido que sale de una casilla y vuelve a ella. En la sala escasa no
# hay muros, asi que este cuadrado siempre se puede recorrer.
CICLO = [(4, 4), (3, 4), (3, 5), (4, 5), (4, 4)]


def extra(a, b, terminal=False) -> float:
    return float(mi_moldeado(a, b, terminal))


def test_la_funcion_dice_algo():
    """La funcion nula no es una respuesta."""
    muestras = [((7, 0), (6, 0), False), ((3, 5), (3, 6), False), ((0, 10), (0, 11), True)]
    assert any(abs(extra(a, b, t)) > 1e-12 for a, b, t in muestras), (
        "Su funcion devuelve cero en todas partes. Eso no es una pista."
    )


def test_no_paga_por_quedarse_quieto():
    """Chocar contra un muro y no moverse no puede dar dinero.

    Si lo da, el agente puede cobrar sin avanzar, y va a preferir eso a
    resolver el problema. Es el fallo mas comun y el mas caro.
    """
    for pos in [(4, 4), (1, 9), (6, 2)]:
        assert extra(pos, pos) <= 1e-12, (
            f"Quedarse en {pos} paga {extra(pos, pos):+.4f}. Con eso, dar vueltas "
            "sin llegar a ninguna parte es rentable."
        )


def test_dar_una_vuelta_completa_no_deja_ganancia():
    """La propiedad que lo decide todo, y se comprueba sin entrenar.

    Si recorrer un ciclo cerrado suma algo positivo, existe una politica que
    gana recompensa infinita sin acercarse nunca a la meta. Y el agente la va a
    encontrar, porque para eso sirve.

    La unica forma conocida de garantizar que esto valga cero es que la
    recompensa extra sea una **diferencia** de una funcion del estado. Ese es el
    resultado de Ng, Harada y Russell (1999), y es lo que se estudia el sabado.
    """
    total = sum(extra(a, b) for a, b in zip(CICLO, CICLO[1:]))
    assert total <= 1e-9, (
        f"Recorrer el cuadrado {CICLO[0]} y volver deja {total:+.4f} de ganancia. "
        "Repitiendolo mil veces, su agente cobra mil veces eso sin llegar a la meta."
    )


def test_con_gamma_uno_el_ciclo_suma_exactamente_cero():
    """Version exigente de la anterior, y la que separa lo correcto de lo que
    simplemente no es rentable todavia."""
    total = sum(extra(a, b) for a, b in zip(CICLO, CICLO[1:]))
    if abs(GAMMA - 1.0) < 1e-9:
        assert total == pytest.approx(0.0, abs=1e-9)
    else:
        # Con gamma < 1 la suma no es exactamente cero, pero tiene que ser
        # pequena comparada con lo que cuesta un paso.
        assert abs(total) < 0.04, (
            f"El ciclo deja {total:+.4f}, comparable al coste de los pasos que lo "
            "recorren. Su pista pesa mas que el problema."
        )


def test_acercarse_vale_mas_que_alejarse():
    """Una pista tiene que informar. Si moverse hacia la meta no vale mas que
    alejarse, su funcion no le esta diciendo nada al agente."""
    hacia = extra((4, 5), (3, 5))
    lejos = extra((4, 5), (5, 5))
    assert hacia > lejos, (
        f"Acercarse paga {hacia:+.4f} y alejarse paga {lejos:+.4f}. "
        "Su pista no distingue una direccion de la otra."
    )
