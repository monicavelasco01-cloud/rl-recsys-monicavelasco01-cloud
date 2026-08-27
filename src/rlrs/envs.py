"""Entornos del curso.

``GridWorld`` es deliberadamente pequeno: cabe entero en la cabeza, permite
calcular a mano lo que el codigo calcula, y expone la misma interfaz que
Gymnasium (``reset`` / ``step``), de modo que lo que se aprende aqui sirve
despues sin traducir nada.
"""

from __future__ import annotations

import numpy as np

# Acciones, en el orden que usaremos todo el curso.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ("arriba", "derecha", "abajo", "izquierda")
ARROWS = ("^", ">", "v", "<")

_DR = (-1, 0, 1, 0)
_DC = (0, 1, 0, -1)


class GridWorld:
    """Mundo de cuadricula con transiciones estocasticas.

    El agente parte de ``start`` y se mueve por una rejilla ``n_rows x n_cols``.
    Con probabilidad ``1 - noise`` avanza en la direccion elegida; el resto de la
    masa se reparte por igual entre las dos direcciones perpendiculares. Chocar
    contra un muro o contra el borde deja al agente donde estaba.

    Parameters
    ----------
    n_rows, n_cols:
        Dimensiones de la rejilla.
    walls:
        Casillas intransitables, como pares ``(fila, columna)``.
    terminals:
        Diccionario ``{(fila, columna): recompensa}``. Entrar en una de estas
        casillas termina el episodio y entrega esa recompensa.
    step_reward:
        Recompensa por cada transicion no terminal. Negativa, para que el agente
        prefiera caminos cortos.
    noise:
        Probabilidad total de desviarse de la accion elegida.
    max_steps:
        Corte de seguridad: por encima de este numero el episodio se trunca.

    Notes
    -----
    La configuracion por defecto es la del widget de iteracion de valor de la
    guia: rejilla 4x5, meta ``+1`` arriba a la derecha y trampa ``-1`` justo
    debajo. Sirve para comparar a mano lo que devuelve el codigo.
    """

    def __init__(
        self,
        n_rows: int = 4,
        n_cols: int = 5,
        walls: set[tuple[int, int]] | None = None,
        terminals: dict[tuple[int, int], float] | None = None,
        start: tuple[int, int] = (3, 0),
        step_reward: float = -0.04,
        noise: float = 0.2,
        max_steps: int = 100,
    ) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"noise debe estar en [0, 1]; se recibio {noise}")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.walls = set(walls) if walls is not None else {(1, 1), (2, 3)}
        self.terminals = dict(terminals) if terminals is not None else {(0, 4): 1.0, (1, 4): -1.0}
        self.start = start
        self.step_reward = step_reward
        self.noise = noise
        self.max_steps = max_steps

        if self.start in self.walls or self.start in self.terminals:
            raise ValueError("La casilla inicial no puede ser un muro ni un estado terminal.")

        self.n_actions = 4
        self.n_states = n_rows * n_cols
        self._rng = np.random.default_rng()
        self._pos = start
        self._t = 0

    # ---------------------------------------------------------------- utilidades

    def state_index(self, pos: tuple[int, int]) -> int:
        """Convierte ``(fila, columna)`` en el indice plano del estado."""
        return pos[0] * self.n_cols + pos[1]

    def state_pos(self, index: int) -> tuple[int, int]:
        """Operacion inversa de :meth:`state_index`."""
        return divmod(index, self.n_cols)

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return pos in self.walls

    def is_terminal(self, pos: tuple[int, int]) -> bool:
        return pos in self.terminals

    def reward_of(self, pos: tuple[int, int]) -> float:
        """Recompensa por *entrar* en ``pos``."""
        return self.terminals.get(pos, self.step_reward)

    def _move(self, pos: tuple[int, int], direction: int) -> tuple[int, int]:
        r, c = pos
        nr, nc = r + _DR[direction], c + _DC[direction]
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            return pos
        if self.is_wall((nr, nc)):
            return pos
        return (nr, nc)

    def transitions(self, pos: tuple[int, int], action: int) -> list[tuple[tuple[int, int], float]]:
        """Distribucion ``P(s' | s, a)`` como lista de ``(estado, probabilidad)``.

        Es el modelo del entorno. La programacion dinamica lo usa; los metodos
        sin modelo (Q-learning, SARSA) deben ignorarlo: solo pueden llamar a
        :meth:`step`.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"accion invalida: {action}")
        outcomes: dict[tuple[int, int], float] = {}
        candidates = (
            (action, 1.0 - self.noise),
            ((action + 1) % 4, self.noise / 2.0),
            ((action + 3) % 4, self.noise / 2.0),
        )
        for direction, prob in candidates:
            if prob <= 0.0:
                continue
            nxt = self._move(pos, direction)
            outcomes[nxt] = outcomes.get(nxt, 0.0) + prob
        return sorted(outcomes.items())

    # ---------------------------------------------------------- interfaz Gymnasium

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        """Reinicia el episodio y devuelve ``(observacion, info)``."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self.start
        self._t = 0
        return self.state_index(self._pos), {"pos": self._pos}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Ejecuta ``action`` y devuelve ``(obs, recompensa, terminado, truncado, info)``."""
        outcomes = self.transitions(self._pos, action)
        states = [s for s, _ in outcomes]
        probs = np.array([p for _, p in outcomes], dtype=float)
        probs /= probs.sum()
        choice = self._rng.choice(len(states), p=probs)
        self._pos = states[choice]
        self._t += 1

        reward = self.reward_of(self._pos)
        terminated = self.is_terminal(self._pos)
        truncated = (not terminated) and self._t >= self.max_steps
        return self.state_index(self._pos), reward, terminated, truncated, {"pos": self._pos}

    # ------------------------------------------------------------------- pintado

    def render_values(self, values: np.ndarray, policy: np.ndarray | None = None) -> str:
        """Devuelve la rejilla como texto, para mirarla sin salir de la terminal."""
        lines = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                pos = (r, c)
                if self.is_wall(pos):
                    cells.append("  ###  ")
                elif self.is_terminal(pos):
                    cells.append(f" {self.terminals[pos]:+.0f}    ")
                else:
                    idx = self.state_index(pos)
                    arrow = ARROWS[int(policy[idx])] if policy is not None else " "
                    cells.append(f"{values[idx]:+.2f}{arrow} ")
            lines.append(" ".join(cells))
        return "\n".join(lines)
