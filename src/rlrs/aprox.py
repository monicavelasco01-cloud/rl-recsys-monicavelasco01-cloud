"""Aproximacion de funciones, y la triada mortal.

Hasta ahora la tabla ``Q`` tenia una casilla por cada par estado-accion. Eso
funciona con veinte casillas y deja de funcionar con cualquier problema real:
en Meridiano hay ciento diez habilidades y mas de cuatro mil secuencias, y la
tabla no cabe. La salida es dejar de guardar un numero por estado y guardar en
su lugar unos pocos **pesos**, con los que el valor de cualquier estado se
calcula.

Eso se paga. Las garantias de convergencia que vimos en la sesion 1 valian para
la tabla y **dejan de valer** aqui. Este modulo no lo cuenta: lo provoca.

Las tres patas de la triada, en los nombres que usa el codigo:

- ``aproximacion="lineal"``      guardar pesos en vez de casillas
- ``semigradiente=True``         apoyarse en la propia estimacion siguiente
- ``fuera_de_politica=True``     aprender de una politica distinta a la que actua

Con las tres a la vez, el metodo puede divergir. Quitando **cualquiera** de las
tres, converge. Eso es exactamente lo que hace ``estrella_de_baird`` y es lo que
el laboratorio pide predecir antes de ejecutar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .envs import GridWorld

Caracteristicas = Callable[[int], np.ndarray]


# ══════════════════════════════════════════════════════════════════════════
# 1 · Como se describe un estado con pocos numeros
# ══════════════════════════════════════════════════════════════════════════

def caracteristicas_tabulares(env: GridWorld) -> Caracteristicas:
    """Una casilla encendida por estado. Es la tabla de siempre, disfrazada.

    Sirve para comprobar algo que conviene tener claro antes de seguir: **la
    tabla es un caso particular de la aproximacion lineal**, el que no
    generaliza nada. Si el codigo nuevo no reproduce el resultado del codigo
    viejo con estas caracteristicas, el codigo nuevo esta mal.
    """
    n = env.n_states
    identidad = np.eye(n, dtype=float)
    return lambda s: identidad[s]


def caracteristicas_posicion(env: GridWorld) -> Caracteristicas:
    """Cuatro numeros por estado: sesgo, fila, columna y cercania a la meta.

    Veinte estados descritos con cuatro pesos por accion. Generaliza mucho, y
    generalizar es justo lo que queremos: lo aprendido en una casilla se
    contagia a sus vecinas. El precio es que dos estados distintos pueden
    volverse indistinguibles, y ahi es donde empiezan los problemas.
    """
    metas = [pos for pos, r in env.terminals.items() if r > 0]
    escala = float(env.n_rows + env.n_cols)

    def phi(s: int) -> np.ndarray:
        f, c = env.state_pos(s)
        d = min(abs(f - m[0]) + abs(c - m[1]) for m in metas) if metas else 0
        return np.array([1.0, f / env.n_rows, c / env.n_cols, 1.0 - d / escala])

    return phi


def caracteristicas_agregadas(env: GridWorld, bloque: int = 2) -> Caracteristicas:
    """Agrupa las casillas en bloques y les da a todas el mismo peso.

    Es la forma mas brusca de generalizar: dos casillas del mismo bloque quedan
    **obligadas** a tener el mismo valor, aunque una sea la meta y la otra la
    trampa. Sirve para ver de que tamano es el error que introduce la
    aproximacion, antes de mezclarlo con el error del aprendizaje.
    """
    filas = (env.n_rows + bloque - 1) // bloque
    cols = (env.n_cols + bloque - 1) // bloque
    n = filas * cols
    identidad = np.eye(n, dtype=float)

    def phi(s: int) -> np.ndarray:
        f, c = env.state_pos(s)
        return identidad[(f // bloque) * cols + (c // bloque)]

    return phi


# ══════════════════════════════════════════════════════════════════════════
# 2 · Control con aproximacion lineal
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AprendizajeAproximado:
    """Lo que devuelve un metodo con aproximacion. Mismo espiritu que ``Aprendizaje``."""

    w: np.ndarray                 # (n_caracteristicas, n_acciones)
    retornos: np.ndarray
    nombre: str
    episodios: int

    def q_de(self, x: np.ndarray) -> np.ndarray:
        """Los valores de las acciones en un estado descrito por ``x``."""
        return x @ self.w

    def tabla_q(self, env: GridWorld, phi: Caracteristicas) -> np.ndarray:
        """Reconstruye la tabla ``Q`` completa, para poder compararla con la de ayer."""
        return np.stack([phi(s) @ self.w for s in range(env.n_states)])

    @property
    def politica_de(self):
        def politica(env: GridWorld, phi: Caracteristicas) -> np.ndarray:
            return self.tabla_q(env, phi).argmax(axis=1)

        return politica


def _epsilon_avida(q_estado: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_estado.size))
    maximos = np.flatnonzero(q_estado == q_estado.max())
    return int(rng.choice(maximos))


def sarsa_semigradiente(
    env: GridWorld,
    phi: Caracteristicas,
    episodes: int = 5000,
    gamma: float = 0.9,
    alpha: float = 0.05,
    epsilon: float = 0.3,
    epsilon_final: float | None = 0.05,
    seed: int = 0,
) -> AprendizajeAproximado:
    """SARSA sobre pesos en vez de casillas.

    La unica diferencia con el SARSA de la sesion 2 es donde se guarda lo
    aprendido. Alli se sumaba ``alpha * delta`` a una casilla; aqui se suma
    ``alpha * delta * x`` a un vector de pesos, y ese ``x`` reparte la
    correccion entre todas las caracteristicas activas. Por eso una
    actualizacion en un estado mueve el valor de otros: eso es generalizar, y es
    lo que queriamos.

    Se llama **semi**gradiente porque al derivar se ignora que el objetivo
    ``r + gamma * q(s', a')`` tambien depende de ``w``. Es una trampa
    deliberada, funciona casi siempre, y es una de las tres patas de la triada.
    """
    rng = np.random.default_rng(seed)
    n_car = phi(0).size
    w = np.zeros((n_car, env.n_actions), dtype=float)
    retornos = np.empty(episodes, dtype=float)

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        fin = epsilon if epsilon_final is None else epsilon_final
        eps = epsilon + frac * (fin - epsilon)

        s, _ = env.reset(seed=int(rng.integers(2**31)))
        x = phi(s)
        a = _epsilon_avida(x @ w, eps, rng)
        total = 0.0

        while True:
            s2, r, term, trunc, _ = env.step(a)
            total += r
            if term:
                delta = r - x @ w[:, a]
                w[:, a] += alpha * delta * x
                break
            x2 = phi(s2)
            a2 = _epsilon_avida(x2 @ w, eps, rng)
            delta = r + gamma * (x2 @ w[:, a2]) - x @ w[:, a]
            w[:, a] += alpha * delta * x
            x, a = x2, a2
            if trunc:
                break

        retornos[ep] = total

    return AprendizajeAproximado(w=w, retornos=retornos, nombre="sarsa-semigradiente", episodios=episodes)


def q_learning_semigradiente(
    env: GridWorld,
    phi: Caracteristicas,
    episodes: int = 5000,
    gamma: float = 0.9,
    alpha: float = 0.05,
    epsilon: float = 0.3,
    epsilon_final: float | None = 0.05,
    seed: int = 0,
) -> AprendizajeAproximado:
    """Q-learning sobre pesos. Fuera de politica y con aproximacion: dos de las tres patas."""
    rng = np.random.default_rng(seed)
    n_car = phi(0).size
    w = np.zeros((n_car, env.n_actions), dtype=float)
    retornos = np.empty(episodes, dtype=float)

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        fin = epsilon if epsilon_final is None else epsilon_final
        eps = epsilon + frac * (fin - epsilon)

        s, _ = env.reset(seed=int(rng.integers(2**31)))
        x = phi(s)
        total = 0.0

        while True:
            a = _epsilon_avida(x @ w, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            total += r
            if term:
                objetivo = r
            else:
                x2 = phi(s2)
                objetivo = r + gamma * float((x2 @ w).max())
            delta = objetivo - float(x @ w[:, a])
            w[:, a] += alpha * delta * x
            if term or trunc:
                break
            x = x2

        retornos[ep] = total

    return AprendizajeAproximado(w=w, retornos=retornos, nombre="q-learning-semigradiente", episodios=episodes)


def error_maximo(tabla_q: np.ndarray, valores_optimos: np.ndarray, env: GridWorld) -> float:
    """Distancia maxima entre ``max_a Q(s,a)`` y el ``V*`` de la sesion 1.

    Misma medida que se uso en la sesion 2, para que las cifras de las tres
    sesiones se puedan poner en la misma tabla sin traducir nada.
    """
    peor = 0.0
    for s in range(env.n_states):
        pos = env.state_pos(s)
        if env.is_wall(pos) or env.is_terminal(pos):
            continue
        peor = max(peor, abs(float(tabla_q[s].max()) - float(valores_optimos[s])))
    return peor


# ══════════════════════════════════════════════════════════════════════════
# 3 · La triada mortal, provocada
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Divergencia:
    """El registro de un experimento de la triada."""

    pesos: np.ndarray          # (pasos + 1, n_pesos)
    norma: np.ndarray          # (pasos + 1,) norma euclidea de los pesos
    etiqueta: str
    diverge: bool

    def __str__(self) -> str:
        veredicto = "DIVERGE" if self.diverge else "converge"
        return f"{self.etiqueta:<34s} {veredicto:>8s}   ‖w‖ final = {self.norma[-1]:,.1f}"


def estrella_de_baird(
    pasos: int = 1000,
    alpha: float = 0.01,
    gamma: float = 0.99,
    aproximacion: Literal["lineal", "tabular"] = "lineal",
    semigradiente: bool = True,
    fuera_de_politica: bool = True,
    seed: int = 0,
) -> Divergencia:
    """El contraejemplo de Baird, con interruptor para cada pata de la triada.

    El problema es de juguete y esta elegido a proposito para que no haya donde
    esconderse. Siete estados, dos acciones, **todas las recompensas valen
    cero**. El valor verdadero de todos los estados es cero, y cualquier metodo
    razonable deberia acercarse a cero. Con las tres patas puestas, los pesos se
    van al infinito.

    Las dos acciones son las clasicas de Baird: la *punteada* lleva a uno de los
    seis estados de arriba al azar, la *solida* lleva siempre al de abajo. La
    politica que actua elige punteada seis de cada siete veces; la politica que
    se evalua elige solida siempre. De ahi sale el ``fuera de politica``.

    Parameters
    ----------
    aproximacion:
        ``"lineal"`` usa las 8 caracteristicas solapadas del contraejemplo, en
        las que ningun estado tiene su propio peso. ``"tabular"`` le da a cada
        estado el suyo, que es quitar la primera pata.
    semigradiente:
        ``True`` ignora la dependencia del objetivo respecto a ``w``, como hacen
        todos los metodos que hemos visto. ``False`` usa el gradiente completo
        del error de Bellman, que es quitar la segunda pata.
    fuera_de_politica:
        ``True`` corrige con el cociente de importancia entre las dos politicas.
        ``False`` evalua la politica que de verdad se esta ejecutando, que es
        quitar la tercera pata.

    Returns
    -------
    Un ``Divergencia`` con la trayectoria de los pesos, para poder graficarla.

    References
    ----------
    Baird (1995), y Sutton y Barto (2018), seccion 11.2.
    """
    rng = np.random.default_rng(seed)
    n_estados = 7
    ARRIBA, ABAJO = range(6), 6
    PUNTEADA, SOLIDA = 0, 1

    if aproximacion == "lineal":
        # Las caracteristicas del contraejemplo: v(s_i) = 2*w_i + w_7 para los
        # seis de arriba, y v(s_6) = w_6 + 2*w_7 para el de abajo. Ningun estado
        # tiene un peso propio y exclusivo: por eso no puede representar
        # cualquier funcion, y por eso se puede romper.
        X = np.zeros((n_estados, 8), dtype=float)
        for i in ARRIBA:
            X[i, i] = 2.0
            X[i, 7] = 1.0
        X[ABAJO, 6] = 1.0
        X[ABAJO, 7] = 2.0
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    elif aproximacion == "tabular":
        X = np.eye(n_estados, dtype=float)
        w = np.ones(n_estados, dtype=float)
        w[ABAJO] = 10.0
    else:
        raise ValueError(f"aproximacion desconocida: {aproximacion!r}")

    # Politica que actua y politica que se evalua.
    b = np.array([6.0 / 7.0, 1.0 / 7.0])
    pi = np.array([0.0, 1.0]) if fuera_de_politica else b

    historia = np.empty((pasos + 1, w.size), dtype=float)
    historia[0] = w

    for t in range(pasos):
        s = int(rng.integers(n_estados))
        a = PUNTEADA if rng.random() < b[PUNTEADA] else SOLIDA
        s2 = int(rng.choice(list(ARRIBA))) if a == PUNTEADA else ABAJO

        rho = pi[a] / b[a]
        v_s = float(X[s] @ w)
        v_s2 = float(X[s2] @ w)
        delta = 0.0 + gamma * v_s2 - v_s          # todas las recompensas son cero

        if semigradiente:
            grad = X[s]
        else:
            # Gradiente completo del error de Bellman: el objetivo tambien
            # depende de w, y aqui si se deriva.
            grad = X[s] - gamma * X[s2]

        w = w + alpha * rho * delta * grad
        historia[t + 1] = w

        if not np.isfinite(w).all():
            historia[t + 1 :] = w
            break

    norma = np.linalg.norm(historia, axis=1)
    diverge = bool(not np.isfinite(norma[-1]) or norma[-1] > 10.0 * norma[0])

    partes = [
        "lineal" if aproximacion == "lineal" else "tabular",
        "semigradiente" if semigradiente else "gradiente completo",
        "fuera de politica" if fuera_de_politica else "dentro de politica",
    ]
    return Divergencia(pesos=historia, norma=norma, etiqueta=" · ".join(partes), diverge=diverge)


def triada_completa(pasos: int = 1000, alpha: float = 0.01, seed: int = 0) -> list[Divergencia]:
    """Los cuatro experimentos: las tres patas, y luego quitando una cada vez.

    Es la demostracion por eliminacion. Si al quitar cualquiera de las tres el
    metodo converge, entonces ninguna de las tres es culpable por si sola: lo
    que rompe es la combinacion. Esa frase se puede decir; esta funcion la
    prueba.
    """
    return [
        estrella_de_baird(pasos, alpha, seed=seed),
        estrella_de_baird(pasos, alpha, aproximacion="tabular", seed=seed),
        estrella_de_baird(pasos, alpha, semigradiente=False, seed=seed),
        estrella_de_baird(pasos, alpha, fuera_de_politica=False, seed=seed),
    ]
