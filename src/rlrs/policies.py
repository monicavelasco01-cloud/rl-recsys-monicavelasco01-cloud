"""Politicas.

Una politica es cualquier objeto que sepa responder ``act(obs) -> accion``.
Nada mas. Esa interfaz minima es la que hace que todo el curso sea comparable:
una politica aleatoria, una heuristica, Q-learning tabular o una red neuronal
entrenada con PPO se evaluan con exactamente el mismo codigo.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Contrato que cumple toda politica del curso."""

    name: str

    def act(self, obs: int) -> int:
        """Devuelve la accion a ejecutar en la observacion ``obs``."""
        ...

    def reset(self, seed: int | None = None) -> None:
        """Reinicia el estado interno (por ejemplo, el generador aleatorio)."""
        ...


class RandomPolicy:
    """Elige uniformemente al azar. Es la linea base mas baja posible.

    Sirve para dos cosas: comprobar que el arnes de evaluacion funciona, y
    tener un suelo contra el que medir. Una politica que no le gana a esta no
    ha aprendido nada.
    """

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.name = "aleatoria"
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: int) -> int:  # noqa: ARG002 - la politica aleatoria ignora obs
        return int(self._rng.integers(self.n_actions))

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self._seed if seed is None else seed)


class GreedyTabularPolicy:
    """Politica avida derivada de una tabla de acciones por estado.

    Es lo que produce la iteracion de valor: para cada estado, una accion fija.
    """

    def __init__(self, table: np.ndarray, name: str = "avida") -> None:
        self.table = np.asarray(table, dtype=int)
        if self.table.ndim != 1:
            raise ValueError("La tabla debe ser un vector con una accion por estado.")
        self.name = name

    def act(self, obs: int) -> int:
        return int(self.table[obs])

    def reset(self, seed: int | None = None) -> None:
        """No tiene estado interno; la firma existe para cumplir el contrato."""
        return None
