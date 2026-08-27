"""Arnes de evaluacion.

Una sola corrida de un agente no es un resultado: es una anecdota. Todo lo que
midamos en este curso pasa por aqui, con semillas explicitas y con dispersion
reportada. Si un numero no viene con su intervalo, no vale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rlrs.envs import GridWorld
from rlrs.policies import Policy


@dataclass
class EvalResult:
    """Resultado de evaluar una politica sobre varios episodios y semillas."""

    policy_name: str
    returns: np.ndarray = field(repr=False)
    lengths: np.ndarray = field(repr=False)
    success_rate: float
    seeds: tuple[int, ...] = ()

    @property
    def mean(self) -> float:
        return float(self.returns.mean())

    @property
    def std(self) -> float:
        return float(self.returns.std(ddof=1)) if self.returns.size > 1 else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        """Intervalo de confianza aproximado al 95 % para la media."""
        if self.returns.size < 2:
            return (self.mean, self.mean)
        half = 1.96 * self.std / np.sqrt(self.returns.size)
        return (self.mean - half, self.mean + half)

    def __str__(self) -> str:
        low, high = self.ci95
        return (
            f"{self.policy_name:<14} retorno {self.mean:+.3f} "
            f"[{low:+.3f}, {high:+.3f}]  exito {self.success_rate:5.1%}  "
            f"pasos {self.lengths.mean():5.1f}"
        )


def run_episode(env: GridWorld, policy: Policy, seed: int) -> tuple[float, int, bool]:
    """Ejecuta un episodio y devuelve ``(retorno, pasos, exito)``.

    El retorno es la suma sin descontar de las recompensas: es lo que de verdad
    obtuvo el agente. El descuento ``gamma`` es una herramienta del algoritmo,
    no la metrica del problema, y confundirlos es un error frecuente.
    """
    obs, _ = env.reset(seed=seed)
    policy.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            success = terminated and env.terminals.get(info["pos"], 0.0) > 0
            return total, steps, bool(success)


def evaluate(
    env: GridWorld,
    policy: Policy,
    episodes: int = 200,
    base_seed: int = 0,
) -> EvalResult:
    """Evalua ``policy`` sobre ``episodes`` episodios con semillas reproducibles.

    Las semillas son ``base_seed, base_seed + 1, ...``: la misma llamada da
    siempre el mismo resultado, y dos politicas evaluadas con el mismo
    ``base_seed`` se enfrentan exactamente a los mismos episodios.
    """
    if episodes < 1:
        raise ValueError("episodes debe ser al menos 1")

    seeds = tuple(base_seed + i for i in range(episodes))
    returns = np.empty(episodes, dtype=float)
    lengths = np.empty(episodes, dtype=int)
    successes = 0

    for i, seed in enumerate(seeds):
        total, steps, ok = run_episode(env, policy, seed)
        returns[i] = total
        lengths[i] = steps
        successes += int(ok)

    return EvalResult(
        policy_name=policy.name,
        returns=returns,
        lengths=lengths,
        success_rate=successes / episodes,
        seeds=seeds,
    )


def compare(env: GridWorld, policies: list[Policy], episodes: int = 200, base_seed: int = 0) -> list[EvalResult]:
    """Evalua varias politicas sobre los mismos episodios y ordena por retorno."""
    results = [evaluate(env, p, episodes=episodes, base_seed=base_seed) for p in policies]
    return sorted(results, key=lambda r: r.mean, reverse=True)
