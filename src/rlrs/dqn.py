"""DQN, y las dos piezas de ingenieria que lo hacen funcionar.

La sesion 3 termino con un problema abierto: aproximacion de funciones,
bootstrapping y aprendizaje fuera de politica juntos pueden divergir, y eso no
es un caso patologico de laboratorio. **Q-learning con una red neuronal es
exactamente esa combinacion**, las tres patas a la vez.

DQN no elimina ninguna de las tres. Lo que hace es anadir dos piezas que
estabilizan el entrenamiento lo suficiente como para que en la practica
funcione:

- **Repeticion de experiencia.** Las transiciones se guardan y se entrenan en
  lotes barajados. Sin esto, las muestras consecutivas de un episodio estan muy
  correlacionadas y la red se ajusta una y otra vez a la parte del mundo en la
  que el agente acaba de estar.
- **Red objetivo.** El objetivo ``r + gamma * max_a' Q(s', a')`` se calcula con
  una copia congelada de la red, que se refresca cada cierto numero de pasos.
  Sin esto, el objetivo se mueve cada vez que se actualizan los pesos, y se
  esta persiguiendo un blanco que huye.

Las dos se pueden apagar con un interruptor, y apagarlas se nota. Ese es el
experimento de la sesion 4 y esta en ``experiments/profundo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .aprox import Caracteristicas
from .envs import GridWorld
from .redes import Adam, Red


@dataclass
class Memoria:
    """El almacen de transiciones, con reemplazo circular.

    Guarda tuplas ``(s, a, r, s', terminado)`` y devuelve lotes al azar. Es una
    estructura sencilla y es media mitad de DQN.
    """

    capacidad: int
    n_caracteristicas: int

    def __post_init__(self) -> None:
        n, d = self.capacidad, self.n_caracteristicas
        self.s = np.zeros((n, d), dtype=float)
        self.a = np.zeros(n, dtype=int)
        self.r = np.zeros(n, dtype=float)
        self.s2 = np.zeros((n, d), dtype=float)
        self.fin = np.zeros(n, dtype=bool)
        self._i = 0
        self._llena = False

    def __len__(self) -> int:
        return self.capacidad if self._llena else self._i

    def guardar(self, s, a, r, s2, fin) -> None:
        i = self._i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.fin[i] = s, a, r, s2, fin
        self._i = (i + 1) % self.capacidad
        if self._i == 0:
            self._llena = True

    def lote(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self), size=n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.fin[idx]


@dataclass
class Entrenamiento:
    """Lo que devuelve un entrenamiento profundo."""

    red: Red
    retornos: np.ndarray
    normas: np.ndarray          # norma de los pesos, para detectar divergencia
    nombre: str

    def q_de(self, x: np.ndarray) -> np.ndarray:
        return self.red.adelante(x)[0]

    def tabla_q(self, env: GridWorld, phi: Caracteristicas) -> np.ndarray:
        X = np.stack([phi(s) for s in range(env.n_states)])
        return self.red.adelante(X)

    def politica(self, env: GridWorld, phi: Caracteristicas) -> np.ndarray:
        return self.tabla_q(env, phi).argmax(axis=1)

    @property
    def diverge(self) -> bool:
        return bool(not np.isfinite(self.normas[-1]) or self.normas[-1] > 20.0 * self.normas[0])

    def __str__(self) -> str:
        estado = "DIVERGE" if self.diverge else "estable"
        return (f"{self.nombre:<28s} {estado:>8s}   retorno final "
                f"{float(np.mean(self.retornos[-100:])):+.4f}   ‖w‖ {self.normas[-1]:,.1f}")


def dqn(
    env: GridWorld,
    phi: Caracteristicas,
    episodes: int = 600,
    gamma: float = 0.99,
    lr: float = 1e-3,
    oculta: int = 64,
    epsilon: float = 0.3,
    epsilon_final: float = 0.05,
    memoria: int = 5000,
    lote: int = 32,
    refresco_objetivo: int = 200,
    usar_memoria: bool = True,
    usar_red_objetivo: bool = True,
    seed: int = 0,
    nombre: str | None = None,
) -> Entrenamiento:
    """Q-learning con una red neuronal en lugar de una tabla.

    Parameters
    ----------
    usar_memoria:
        ``False`` entrena con la transicion que acaba de ocurrir, una cada vez.
        Es Q-learning con red, sin mas, y es la version que se rompe.
    usar_red_objetivo:
        ``False`` calcula el objetivo con la misma red que se esta
        actualizando. Es la otra pieza que se puede quitar para ver que pasa.

    Notes
    -----
    Todo lo demas es identico a lo que se hizo en la sesion 2: politica
    epsilon-avida para actuar, objetivo ``r + gamma * max`` para aprender. Lo
    unico que cambia es donde se guardan los valores. Merece la pena repetirlo
    porque es facil creer que DQN es un algoritmo nuevo, y no lo es: es el
    mismo algoritmo con otro almacen y dos muletas.
    """
    rng = np.random.default_rng(seed)
    d = phi(0).size
    red = Red((d, oculta, env.n_actions), seed=seed)
    objetivo = Red((d, oculta, env.n_actions), seed=seed)
    objetivo.copiar_de(red)
    opt = Adam(red.parametros(), lr=lr)
    mem = Memoria(memoria, d)

    retornos = np.empty(episodes, dtype=float)
    normas = np.empty(episodes, dtype=float)
    pasos = 0

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        eps = epsilon + frac * (epsilon_final - epsilon)
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        x = phi(s)
        total = 0.0

        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                q = red.adelante(x)[0]
                a = int(rng.choice(np.flatnonzero(q == q.max())))

            s2, r, term, trunc, _ = env.step(a)
            x2 = phi(s2)
            total += r
            mem.guardar(x, a, r, x2, term)
            pasos += 1

            # ── el paso de aprendizaje ────────────────────────────────────
            if usar_memoria:
                if len(mem) >= lote:
                    bs, ba, br, bs2, bfin = mem.lote(lote, rng)
                else:
                    bs = ba = None
            else:
                bs = x[None, :]
                ba = np.array([a])
                br = np.array([r])
                bs2 = x2[None, :]
                bfin = np.array([term])

            if bs is not None:
                red_objetivo = objetivo if usar_red_objetivo else red
                q_siguiente = red_objetivo.adelante(bs2).max(axis=1)
                objetivos = br + gamma * q_siguiente * (~bfin)

                q = red.adelante(bs)
                filas = np.arange(len(ba))
                # Solo la accion tomada recibe gradiente. Las otras salidas de
                # la red no se tocan, porque de ellas no hay dato nuevo.
                grad = np.zeros_like(q)
                grad[filas, ba] = 2.0 * (q[filas, ba] - objetivos) / len(ba)
                red.atras(grad)
                opt.paso(red.parametros(), red.gradientes())

            if usar_red_objetivo and pasos % refresco_objetivo == 0:
                objetivo.copiar_de(red)

            x = x2
            if term or trunc:
                break

        retornos[ep] = total
        normas[ep] = red.norma()

    if nombre is None:
        piezas = []
        piezas.append("con memoria" if usar_memoria else "SIN memoria")
        piezas.append("con red objetivo" if usar_red_objetivo else "SIN red objetivo")
        nombre = " · ".join(piezas)

    return Entrenamiento(red=red, retornos=retornos, normas=normas, nombre=nombre)


def ablacion_dqn(
    env_fabrica: Callable[[], GridWorld],
    phi_fabrica: Callable[[GridWorld], Caracteristicas],
    episodes: int = 600,
    seed: int = 0,
    **kw,
) -> list[Entrenamiento]:
    """DQN completo y las tres versiones a las que les falta algo.

    Es la demostracion por eliminacion de la sesion 4, igual que la de la
    triada lo fue de la sesion 3. Si al quitar una pieza el entrenamiento se
    deteriora, esa pieza estaba haciendo algo; si no cambia nada, sobra. Las
    dos respuestas son informativas y las dos hay que medirlas.
    """
    combinaciones = [
        (True, True, "DQN completo"),
        (True, False, "sin red objetivo"),
        (False, True, "sin memoria"),
        (False, False, "sin ninguna de las dos"),
    ]
    salida = []
    for con_memoria, con_objetivo, etiqueta in combinaciones:
        env = env_fabrica()
        salida.append(
            dqn(env, phi_fabrica(env), episodes=episodes, seed=seed,
                usar_memoria=con_memoria, usar_red_objetivo=con_objetivo,
                nombre=etiqueta, **kw)
        )
    return salida
