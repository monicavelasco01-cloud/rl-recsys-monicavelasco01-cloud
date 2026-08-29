"""Aprender sin modelo del entorno: Monte Carlo y diferencias temporales.

La programacion dinamica de `rlrs.dp` necesita `env.transitions()`, es decir,
conocer P. Lo que hay aqui **no puede llamar a ese metodo**: solo puede
`reset()` y `step()`, como si el mundo fuera opaco. Esa frontera es el eje del
Bloque 1 y esta escrita en el codigo a proposito.

Tres algoritmos, en el orden en que aparecen en la sesion 2:

  mc_control   Monte Carlo con primera visita. Espera a que el episodio termine
               y reparte el retorno observado. Sin sesgo, con mucha varianza.
  sarsa        Diferencias temporales, dentro de politica. Aprende el valor de
               lo que de verdad hace, exploracion incluida.
  q_learning   Diferencias temporales, fuera de politica. Aprende el valor de
               la politica avida aunque este explorando.

Los tres devuelven la misma estructura, para que compararlos sea trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rlrs.envs import GridWorld


@dataclass
class Aprendizaje:
    """Lo que devuelve un algoritmo sin modelo."""

    q: np.ndarray = field(repr=False)
    retornos: np.ndarray = field(repr=False)
    nombre: str = ""
    episodios: int = 0

    @property
    def politica(self) -> np.ndarray:
        """Accion avida en cada estado, segun la Q aprendida."""
        return self.q.argmax(axis=1).astype(int)

    def media_final(self, ventana: int = 200) -> float:
        """Retorno medio de los ultimos episodios de entrenamiento."""
        n = min(ventana, self.retornos.size)
        return float(self.retornos[-n:].mean())

    def __str__(self) -> str:
        return (
            f"{self.nombre:<12} {self.episodios:>6} episodios  "
            f"retorno final {self.media_final():+.3f}"
        )


def _epsilon_avida(q_estado: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Elige una accion: la mejor con probabilidad 1 - epsilon, al azar el resto."""
    if rng.random() < epsilon:
        return int(rng.integers(q_estado.size))
    # El desempate al azar importa: con Q inicializada a cero, argmax elegiria
    # siempre la accion 0 y el agente saldria sesgado hacia arriba.
    mejores = np.flatnonzero(q_estado == q_estado.max())
    return int(rng.choice(mejores))


def _epsilon_de(episodio: int, total: int, inicial: float, final: float) -> float:
    """Decaimiento lineal de epsilon a lo largo del entrenamiento."""
    if total <= 1:
        return final
    t = episodio / (total - 1)
    return inicial + (final - inicial) * t


def mc_control(
    env: GridWorld,
    episodes: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    epsilon_final: float | None = 0.05,
    seed: int = 0,
) -> Aprendizaje:
    """Control Monte Carlo de primera visita, con politica epsilon-avida.

    Notes
    -----
    No hay tasa de aprendizaje: cada Q(s,a) es la media de todos los retornos
    observados desde ese par. Eso lo hace insesgado, y tambien lento, porque no
    aprende nada hasta que el episodio termina.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    visitas = np.zeros_like(q)
    retornos = np.empty(episodes, dtype=float)

    for ep in range(episodes):
        eps = _epsilon_de(ep, episodes, epsilon, epsilon if epsilon_final is None else epsilon_final)
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        trayectoria: list[tuple[int, int, float]] = []
        total = 0.0

        while True:
            a = _epsilon_avida(q[obs], eps, rng)
            sig, r, term, trunc, _ = env.step(a)
            trayectoria.append((obs, a, r))
            total += r
            obs = sig
            if term or trunc:
                break

        retornos[ep] = total

        # Primera visita: se recorre al reves y solo cuenta la primera aparicion.
        vistos: set[tuple[int, int]] = set()
        g = 0.0
        for s, a, r in reversed(trayectoria):
            g = r + gamma * g
            if (s, a) in vistos:
                continue
            vistos.add((s, a))
            visitas[s, a] += 1
            q[s, a] += (g - q[s, a]) / visitas[s, a]

    return Aprendizaje(q=q, retornos=retornos, nombre="monte-carlo", episodios=episodes)


def sarsa(
    env: GridWorld,
    episodes: int = 5000,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.3,
    epsilon_final: float | None = 0.05,
    seed: int = 0,
) -> Aprendizaje:
    """SARSA: diferencias temporales dentro de politica.

    La actualizacion usa la accion que el agente **va a tomar de verdad**, con
    su exploracion incluida. Por eso aprende el valor de comportarse como se
    esta comportando, y por eso evita los caminos donde explorar sale caro.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    retornos = np.empty(episodes, dtype=float)

    for ep in range(episodes):
        eps = _epsilon_de(ep, episodes, epsilon, epsilon if epsilon_final is None else epsilon_final)
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        a = _epsilon_avida(q[s], eps, rng)
        total = 0.0

        while True:
            s2, r, term, trunc, _ = env.step(a)
            total += r
            if term:
                q[s, a] += alpha * (r - q[s, a])
                break
            a2 = _epsilon_avida(q[s2], eps, rng)
            q[s, a] += alpha * (r + gamma * q[s2, a2] - q[s, a])
            s, a = s2, a2
            if trunc:
                break

        retornos[ep] = total

    return Aprendizaje(q=q, retornos=retornos, nombre="sarsa", episodios=episodes)


def q_learning(
    env: GridWorld,
    episodes: int = 5000,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.3,
    epsilon_final: float | None = 0.05,
    seed: int = 0,
) -> Aprendizaje:
    """Q-learning: diferencias temporales fuera de politica.

    La actualizacion usa el **maximo** sobre las acciones del estado siguiente,
    no la que el agente vaya a tomar. Aprende el valor de la politica avida
    mientras se comporta de otra manera, y por eso converge a la solucion
    optima aunque nunca la ejecute durante el entrenamiento.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    retornos = np.empty(episodes, dtype=float)

    for ep in range(episodes):
        eps = _epsilon_de(ep, episodes, epsilon, epsilon if epsilon_final is None else epsilon_final)
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        total = 0.0

        while True:
            a = _epsilon_avida(q[s], eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            total += r
            objetivo = r if term else r + gamma * q[s2].max()
            q[s, a] += alpha * (objetivo - q[s, a])
            s = s2
            if term or trunc:
                break

        retornos[ep] = total

    return Aprendizaje(q=q, retornos=retornos, nombre="q-learning", episodios=episodes)


def error_frente_a(q: np.ndarray, valores_optimos: np.ndarray, env: GridWorld) -> float:
    """Distancia maxima entre max_a Q(s,a) y el V* que dio la programacion dinamica.

    Es la medida honesta de cuanto le falta a un metodo sin modelo para llegar
    donde ya sabemos que esta la respuesta. Solo tiene sentido en un problema
    pequeno como este, y por eso el laboratorio de la sesion 2 lo usa.
    """
    peor = 0.0
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos = (r, c)
            if env.is_wall(pos) or env.is_terminal(pos):
                continue
            i = env.state_index(pos)
            peor = max(peor, abs(q[i].max() - valores_optimos[i]))
    return float(peor)
