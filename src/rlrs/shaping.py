"""Diseno de recompensa: moldeado, y las dos formas de hacerlo.

La recompensa no viene con el problema: la escribe alguien. Este modulo existe
para que ese "alguien" vea, con numeros, la diferencia entre las dos maneras de
darle pistas a un agente.

La tentacion es siempre la misma. El agente tarda mucho en encontrar la meta,
asi que se le anade un premio por acercarse. Es razonable, es lo que haria
cualquiera, y **cambia la solucion optima del problema**. Este modulo lo
demuestra en vez de afirmarlo.

La alternativa correcta se conoce desde Ng, Harada y Russell (1999): si el
premio extra se escribe como la diferencia de una funcion del estado, y solo
asi, la politica optima no se mueve. Es un resultado con demostracion, no una
recomendacion de estilo.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .envs import GridWorld

Potencial = Callable[[tuple[int, int]], float]


def _pasos_hasta_la_meta(env: GridWorld) -> Callable[[tuple[int, int]], int]:
    metas = [pos for pos, r in env.terminals.items() if r > 0]
    if not metas:
        raise ValueError("El entorno no tiene ninguna casilla terminal con recompensa positiva.")

    def d(pos: tuple[int, int]) -> int:
        return min(abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in metas)

    return d


def cercania_a_meta(env: GridWorld) -> Potencial:
    """Cuanto de cerca esta el agente de la meta, entre 0 y 1.

    Vale 1 en la meta y va bajando segun ``1 / (1 + pasos que faltan)``. Es la
    pista mas obvia que se le puede dar a un agente de rejilla y la que sale
    sola cuando uno improvisa: *dale puntos por estar cerca*.

    Los dos modos de moldeado usan **esta misma funcion**. La diferencia entre
    que funcione y que sea un desastre no esta en la pista, esta en la forma de
    darla.
    """
    d = _pasos_hasta_la_meta(env)
    return lambda pos: 1.0 / (1.0 + d(pos))


def distancia_a_meta(env: GridWorld) -> Potencial:
    """Menos los pasos que faltan hasta la meta. Alternativa a ``cercania_a_meta``.

    Se deja disponible porque en el laboratorio se pide comprobar si la
    conclusion depende de la pista elegida o de la forma de darla. Depende de la
    forma: esta es la comprobacion.
    """
    d = _pasos_hasta_la_meta(env)
    return lambda pos: -float(d(pos))


class GridWorldMoldeado:
    """Un ``GridWorld`` con recompensa modificada, envuelto sin tocar el original.

    Expone la misma interfaz (``reset`` / ``step``) que el entorno de siempre,
    de modo que los algoritmos de ``rlrs.td`` funcionan sobre el sin cambiar una
    linea. Eso es deliberado: el punto de la sesion es que **el algoritmo no se
    entera** de que le cambiaron la recompensa. El que se entera es quien lee
    los resultados, y solo si mide.

    Parameters
    ----------
    env:
        El entorno original. No se modifica.
    modo:
        ``"ingenuo"`` suma ``escala * phi(s')`` en cada paso: un premio por
        **estar** cerca de la meta. Es lo que sale solo cuando uno improvisa, y
        se puede cobrar una y otra vez sin avanzar.
        ``"potencial"`` suma ``escala * (gamma * phi(s') - phi(s))``: un premio
        por **acercarse**, que es cosa distinta. Con esta forma, y solo con
        esta, la politica optima esta demostrado que no cambia.
    phi:
        La funcion de potencial. Por defecto, la cercania a la meta.
    gamma:
        Tiene que ser el mismo descuento con el que se va a entrenar. Si no
        coincide, la garantia del moldeado por potencial no aplica.

    Notes
    -----
    El atajo mental util: el moldeado ingenuo paga por **estar** en un sitio
    bueno, y estar en un sitio se puede repetir indefinidamente. El moldeado por
    potencial paga por **moverse** hacia un sitio bueno, y como es una
    diferencia, la suma a lo largo de un ciclo cerrado es cero. Ahi esta toda la
    demostracion, y por eso el agente no puede hacer trampa con la segunda.
    """

    def __init__(
        self,
        env: GridWorld,
        modo: Literal["ingenuo", "potencial"] = "ingenuo",
        phi: Potencial | None = None,
        gamma: float = 0.9,
        escala: float = 0.5,
    ) -> None:
        if modo not in ("ingenuo", "potencial"):
            raise ValueError(f"modo desconocido: {modo!r}. Use 'ingenuo' o 'potencial'.")
        self.env = env
        self.modo = modo
        self.phi = phi if phi is not None else cercania_a_meta(env)
        self.gamma = gamma
        self.escala = escala

        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_rows, self.n_cols = env.n_rows, env.n_cols

        self._pos_anterior: tuple[int, int] | None = None

    # ── interfaz de entorno ────────────────────────────────────────────────
    def state_index(self, pos: tuple[int, int]) -> int:
        return self.env.state_index(pos)

    def state_pos(self, index: int) -> tuple[int, int]:
        return self.env.state_pos(index)

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return self.env.is_wall(pos)

    def is_terminal(self, pos: tuple[int, int]) -> bool:
        return self.env.is_terminal(pos)

    @property
    def terminals(self) -> dict[tuple[int, int], float]:
        return self.env.terminals

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        s, info = self.env.reset(seed=seed)
        self._pos_anterior = self.env.state_pos(s)
        return s, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        anterior = self._pos_anterior
        s2, r, term, trunc, info = self.env.step(action)
        siguiente = self.env.state_pos(s2)

        extra = self.extra(anterior, siguiente, term)
        self._pos_anterior = siguiente

        info = dict(info)
        info["recompensa_original"] = r
        info["moldeado"] = extra
        return s2, r + extra, term, trunc, info

    # ── la cuenta, aislada para poder probarla ─────────────────────────────
    def extra(
        self,
        anterior: tuple[int, int] | None,
        siguiente: tuple[int, int],
        terminal: bool = False,
    ) -> float:
        """La recompensa extra de una transicion, sin ejecutar nada.

        Se expone aparte porque es la unica linea que de verdad importa de este
        modulo, y porque asi se puede comprobar a mano en el laboratorio.
        """
        if self.modo == "ingenuo":
            return self.escala * self.phi(siguiente)

        # Convenio estandar: el potencial de un estado terminal es cero. Sin
        # esto, el moldeado por potencial deja de ser una diferencia telescopica
        # y la garantia se pierde justo en el ultimo paso del episodio.
        phi_siguiente = 0.0 if terminal else self.phi(siguiente)
        phi_anterior = 0.0 if anterior is None else self.phi(anterior)
        return self.escala * (self.gamma * phi_siguiente - phi_anterior)


class GridWorldConMoldeado:
    """Un ``GridWorld`` con la recompensa extra que escriba **usted**.

    Es la misma envoltura de arriba, pero en vez de elegir entre dos modos
    prefabricados recibe una funcion cualquiera. Es lo que usa el reto de la
    sesion 3: el estudiante escribe la funcion, esta clase la conecta al
    entorno, y los algoritmos de siempre entrenan sin enterarse.

    La funcion recibe tres cosas y devuelve un numero:

    - ``anterior``: la casilla en la que estaba, como ``(fila, columna)``.
    - ``siguiente``: la casilla a la que acaba de llegar.
    - ``terminal``: si ``siguiente`` termina el episodio.
    """

    def __init__(self, env: GridWorld, funcion: Callable[..., float]) -> None:
        self.env = env
        self.funcion = funcion
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_rows, self.n_cols = env.n_rows, env.n_cols
        self._pos_anterior: tuple[int, int] | None = None

    def state_index(self, pos: tuple[int, int]) -> int:
        return self.env.state_index(pos)

    def state_pos(self, index: int) -> tuple[int, int]:
        return self.env.state_pos(index)

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return self.env.is_wall(pos)

    def is_terminal(self, pos: tuple[int, int]) -> bool:
        return self.env.is_terminal(pos)

    @property
    def terminals(self) -> dict[tuple[int, int], float]:
        return self.env.terminals

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        s, info = self.env.reset(seed=seed)
        self._pos_anterior = self.env.state_pos(s)
        return s, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        anterior = self._pos_anterior
        s2, r, term, trunc, info = self.env.step(action)
        siguiente = self.env.state_pos(s2)
        extra = float(self.funcion(anterior, siguiente, term))
        self._pos_anterior = siguiente
        info = dict(info)
        info["recompensa_original"] = r
        info["moldeado"] = extra
        return s2, r + extra, term, trunc, info


def retorno_verdadero(
    env: GridWorld,
    politica: np.ndarray,
    episodios: int = 200,
    base_seed: int = 0,
    max_pasos: int = 200,
) -> tuple[float, float, float]:
    """Retorno de una politica medido con la recompensa **original**.

    Este es el detalle que decide la sesion. Un agente entrenado con recompensa
    moldeada obtiene numeros excelentes... en la recompensa moldeada. Para saber
    si de verdad resolvio el problema hay que medirlo con la recompensa que
    nos importaba antes de empezar a ayudarle.

    Returns
    -------
    Media, y los dos extremos del intervalo de confianza del 95 %.
    """
    rng = np.random.default_rng(base_seed)
    totales = np.empty(episodios, dtype=float)

    for i in range(episodios):
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        total = 0.0
        for _ in range(max_pasos):
            s, r, term, trunc, _ = env.step(int(politica[s]))
            total += r
            if term or trunc:
                break
        totales[i] = total

    media = float(totales.mean())
    if episodios > 1:
        margen = 1.96 * float(totales.std(ddof=1)) / np.sqrt(episodios)
    else:
        margen = 0.0
    return media, media - margen, media + margen
