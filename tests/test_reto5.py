"""Pruebas del contrato del reto 5.

Comprueban la forma de la respuesta, no su calidad: si la politica gana lo
decide ``scripts/reto5.py``, que la mide sobre Meridiano con varias semillas.
"""

from __future__ import annotations

import numpy as np
import pytest

from retos.reto5 import mi_politica

from rlrs.bandidos import Corrida, bandido_dificil, epsilon_avida


def _por_defecto() -> bool:
    """True mientras la respuesta sea la que venia de fabrica."""
    b = bandido_dificil(20, 0.05, seed=0)
    mia = mi_politica(b, 60, 0)
    base = epsilon_avida(b, 60, epsilon=0.0, seed=0)
    return bool(np.array_equal(mia.brazos, base.brazos))


pytestmark = pytest.mark.skipif(
    _por_defecto(),
    reason="El reto 5 todavia no esta contestado: modifica 'mi_politica' en retos/reto5.py",
)


def entorno():
    return bandido_dificil(20, 0.05, seed=0)


def test_devuelve_una_corrida():
    r = mi_politica(entorno(), 50, 0)
    assert isinstance(r, Corrida)


def test_juega_exactamente_las_rondas_que_le_dan():
    """Jugar de mas no es una politica mejor: es otro presupuesto."""
    r = mi_politica(entorno(), 137, 0)
    assert r.rondas == 137, (
        f"Jugaste {r.rondas} rondas y te habian dado 137. El arnes compara todas "
        "las politicas con el mismo presupuesto, asi que esto invalida la medida."
    )


def test_respeta_la_semilla():
    a = mi_politica(entorno(), 80, 7)
    b = mi_politica(entorno(), 80, 7)
    assert np.array_equal(a.brazos, b.brazos), (
        "Dos llamadas con la misma semilla eligieron brazos distintos. Sin eso, "
        "nada de lo que midas es reproducible y el intervalo que reportes es humo."
    )


def test_solo_elige_brazos_que_existen():
    b = entorno()
    r = mi_politica(b, 80, 0)
    assert r.brazos.min() >= 0 and r.brazos.max() < b.n_brazos, (
        f"Elegiste un brazo fuera del rango 0..{b.n_brazos - 1}."
    )


def test_reparte_algo_las_tiradas():
    """El sintoma de la avida pura, comprobado sobre el bandido de libro."""
    b = entorno()
    r = mi_politica(b, 400, 0)
    usados = int((r.reparto(b.n_brazos) > 0).sum())
    assert usados >= 3, (
        f"En 400 rondas tocaste {usados} brazos de {b.n_brazos}. Una politica que "
        "no reparte no se entera de que el brazo que eligio se agoto."
    )
