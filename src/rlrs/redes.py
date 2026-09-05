"""Una red neuronal pequena, escrita a mano, para poder mirarla por dentro.

Este modulo existe por una razon pedagogica y conviene decirla antes de la
primera linea de codigo: **en el bloque 3 vamos a usar PyTorch**, y PyTorch
calcula los gradientes solo. Eso es estupendo para trabajar y es malo para
aprender, porque el paso que hay que entender queda escondido detras de una
llamada a ``.backward()``.

Aqui esta ese paso, escrito. Son dos capas, una funcion de activacion y la
regla de la cadena aplicada a mano. Nada mas. Cuando el sabado veamos la
version en PyTorch y ocupe cuarenta lineas en vez de doscientas, la pregunta
"y esto que hace por dentro" ya va a tener respuesta.

La red es deliberadamente pequena. No hace falta mas para una cuadricula, y
todo lo que se mide en la guia sale de ejecutar esto, no de una biblioteca.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Capa:
    """Una capa densa: ``y = x @ W + b``.

    Guarda la entrada de la ultima pasada hacia adelante porque el gradiente
    respecto a ``W`` la necesita. Es la unica razon por la que una capa tiene
    memoria.
    """

    W: np.ndarray
    b: np.ndarray
    _x: np.ndarray | None = field(default=None, repr=False)
    dW: np.ndarray | None = field(default=None, repr=False)
    db: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def nueva(cls, entradas: int, salidas: int, rng: np.random.Generator) -> "Capa":
        # Inicializacion de He, que es la que corresponde a ReLU: la varianza
        # se escoge para que la senal no se apague ni explote al atravesar
        # varias capas. Con dos capas casi da igual; con veinte, no.
        escala = np.sqrt(2.0 / entradas)
        return cls(W=rng.normal(0.0, escala, size=(entradas, salidas)), b=np.zeros(salidas))

    def adelante(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def atras(self, grad: np.ndarray) -> np.ndarray:
        """Recibe dL/dy y devuelve dL/dx, guardando de paso dL/dW y dL/db."""
        assert self._x is not None, "hay que llamar a adelante() antes que a atras()"
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T


class Red:
    """Perceptron multicapa con activacion ReLU en las capas ocultas.

    Parameters
    ----------
    tamanos:
        Numero de neuronas de cada capa, empezando por las entradas. Por
        ejemplo ``(4, 32, 4)`` es una red que recibe 4 numeros, tiene una capa
        oculta de 32 y devuelve 4 valores, uno por accion.
    seed:
        Semilla de la inicializacion. Dos redes con la misma semilla empiezan
        identicas, que es lo que hace reproducible todo lo demas.

    Notes
    -----
    No hay activacion en la ultima capa. Para valores de accion eso es lo
    correcto: un valor puede ser negativo y no esta acotado, asi que meterle
    una sigmoide o una ReLU al final seria un error. Es un fallo frecuente y
    silencioso, porque la red sigue entrenando y solo aprende peor.
    """

    def __init__(self, tamanos: tuple[int, ...], seed: int = 0) -> None:
        if len(tamanos) < 2:
            raise ValueError("una red necesita al menos entradas y salidas")
        rng = np.random.default_rng(seed)
        self.tamanos = tuple(tamanos)
        self.capas = [Capa.nueva(a, b, rng) for a, b in zip(tamanos, tamanos[1:])]
        self._mascaras: list[np.ndarray] = []

    # ── uso ────────────────────────────────────────────────────────────────
    def adelante(self, x: np.ndarray) -> np.ndarray:
        """Calcula la salida. ``x`` puede ser un ejemplo o un lote de ejemplos."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        self._mascaras = []
        for i, capa in enumerate(self.capas):
            x = capa.adelante(x)
            if i < len(self.capas) - 1:
                mascara = x > 0.0
                self._mascaras.append(mascara)
                x = x * mascara          # ReLU
        return x

    def atras(self, grad: np.ndarray) -> None:
        """Propaga el gradiente de la perdida hacia atras y lo guarda en cada capa."""
        grad = np.atleast_2d(np.asarray(grad, dtype=float))
        for i in range(len(self.capas) - 1, -1, -1):
            if i < len(self.capas) - 1:
                grad = grad * self._mascaras[i]   # derivada de la ReLU
            grad = self.capas[i].atras(grad)

    # ── parametros ─────────────────────────────────────────────────────────
    def parametros(self) -> list[np.ndarray]:
        return [p for capa in self.capas for p in (capa.W, capa.b)]

    def gradientes(self) -> list[np.ndarray]:
        return [g for capa in self.capas for g in (capa.dW, capa.db)]

    def copiar_de(self, otra: "Red") -> None:
        """Copia los pesos de otra red. Es lo que hace la **red objetivo** de DQN."""
        if self.tamanos != otra.tamanos:
            raise ValueError("las dos redes tienen que tener la misma forma")
        for destino, origen in zip(self.capas, otra.capas):
            destino.W = origen.W.copy()
            destino.b = origen.b.copy()

    def norma(self) -> float:
        """Norma euclidea de todos los pesos juntos. Sirve para detectar divergencia."""
        return float(np.sqrt(sum(float((p ** 2).sum()) for p in self.parametros())))


class Adam:
    """El optimizador que usa todo el mundo, en veinte lineas.

    Guarda dos medias moviles por parametro: una del gradiente y otra de su
    cuadrado. La primera da inercia; la segunda adapta el tamano del paso a
    cada peso por separado, de modo que los pesos con gradientes grandes se
    mueven proporcionalmente menos.

    Se implementa aqui, y no se usa descenso de gradiente a secas, porque con
    descenso simple la diferencia entre que DQN aprenda y no aprenda es la
    tasa de aprendizaje, y buscarla a mano en clase no ensena nada.
    """

    def __init__(self, parametros: list[np.ndarray], lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in parametros]
        self.v = [np.zeros_like(p) for p in parametros]
        self.t = 0

    def paso(self, parametros: list[np.ndarray], gradientes: list[np.ndarray]) -> None:
        """Actualiza los parametros **en el sitio**, sin devolver nada."""
        self.t += 1
        correccion1 = 1.0 - self.b1 ** self.t
        correccion2 = 1.0 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(parametros, gradientes)):
            if g is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g ** 2)
            m_ = self.m[i] / correccion1
            v_ = self.v[i] / correccion2
            p -= self.lr * m_ / (np.sqrt(v_) + self.eps)


def gradiente_numerico(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Derivada aproximada por diferencias centradas, solo para comprobar.

    No se usa para entrenar: se usa en las pruebas para verificar que el
    gradiente escrito a mano es el correcto. Si alguna vez toca modificar
    ``Capa.atras``, esta funcion es la que avisa de que se rompio algo.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        original = x[i]
        x[i] = original + h
        mas = f()
        x[i] = original - h
        menos = f()
        x[i] = original
        grad[i] = (mas - menos) / (2.0 * h)
        it.iternext()
    return grad
