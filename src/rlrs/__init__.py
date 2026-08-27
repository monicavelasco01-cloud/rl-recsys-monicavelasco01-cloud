"""Paquete base del modulo *Aprendizaje por Refuerzo y Sistemas de Recomendacion*.

Regla del curso
---------------
El algoritmo vive en el paquete; el notebook es la bitacora.

Todo lo que se ejecute mas de una vez -- un entorno, una politica, una metrica --
se escribe aqui, con su prueba en ``tests/``. Los notebooks de ``notebooks/``
importan de este paquete, exploran y grafican, pero no definen algoritmos.
"""

from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
from rlrs.evaluation import evaluate
from rlrs.policies import GreedyTabularPolicy, Policy, RandomPolicy

__version__ = "0.1.0"

__all__ = [
    "GridWorld",
    "Policy",
    "RandomPolicy",
    "GreedyTabularPolicy",
    "value_iteration",
    "evaluate",
    "__version__",
]
