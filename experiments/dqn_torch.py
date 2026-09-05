"""El mismo DQN de `rlrs.dqn`, en PyTorch, para comparar.

    uv sync --extra dl
    uv run python experiments/dqn_torch.py

Este archivo existe para una sola lamina de la sesion 4: **que se ahorra con el
marco**. Es el mismo algoritmo, el mismo entorno y los mismos hiperparametros
que ``rlrs.dqn``, escrito con PyTorch. Compara las dos versiones tu mismo.

Lo que desaparece al usar el marco:

  - Las 60 lineas de ``rlrs/redes.py``: capas, ReLU, retropropagacion y Adam.
    Aqui son cuatro lineas de ``nn.Sequential`` y una de ``optim.Adam``.
  - El gradiente escrito a mano. Aqui es ``perdida.backward()``.

Lo que NO desaparece, y conviene verlo: la memoria de repeticion, la red
objetivo, la politica epsilon-avida y el bucle de episodios siguen siendo tuyos.
El marco te quita el calculo, no el diseno.

Aviso honesto sobre los numeros: **las dos versiones no van a dar el mismo
resultado**. Distinta inicializacion de pesos y distinto generador aleatorio.
Lo que tiene que coincidir es la conclusion, no la cifra, y esa es exactamente
la disciplina que llevamos tres sesiones practicando.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    raise SystemExit(
        "\n  Falta PyTorch. Instalalo con:\n\n      uv sync --extra dl\n\n"
        "  Pesa bastante, asi que hazlo antes de la sesion y no durante.\n"
    )

from rlrs.aprox import caracteristicas_posicion
from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
from rlrs.shaping import retorno_verdadero

EPISODIOS = 300
GAMMA = 0.9
LR = 1e-3
OCULTA = 64
LOTE = 32
MEMORIA = 5000
REFRESCO = 200
SEMILLA = 0


def rejilla() -> GridWorld:
    return GridWorld(noise=0.2, step_reward=-0.04)


def main() -> None:
    torch.manual_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)

    env = rejilla()
    phi = caracteristicas_posicion(env)
    d = phi(0).size

    # ── esto es lo que sustituye a todo rlrs/redes.py ─────────────────────
    def crear_red() -> nn.Module:
        return nn.Sequential(nn.Linear(d, OCULTA), nn.ReLU(), nn.Linear(OCULTA, env.n_actions))

    red = crear_red()
    objetivo = crear_red()
    objetivo.load_state_dict(red.state_dict())
    opt = torch.optim.Adam(red.parameters(), lr=LR)
    perdida_fn = nn.MSELoss()

    # memoria de repeticion, igual que en la version de NumPy
    mem_s = np.zeros((MEMORIA, d), dtype=np.float32)
    mem_a = np.zeros(MEMORIA, dtype=np.int64)
    mem_r = np.zeros(MEMORIA, dtype=np.float32)
    mem_s2 = np.zeros((MEMORIA, d), dtype=np.float32)
    mem_f = np.zeros(MEMORIA, dtype=bool)
    escritos, cursor = 0, 0

    pasos = 0
    for ep in range(EPISODIOS):
        frac = ep / max(1, EPISODIOS - 1)
        eps = 0.3 + frac * (0.05 - 0.3)
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        x = phi(s).astype(np.float32)

        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                with torch.no_grad():
                    a = int(red(torch.from_numpy(x)).argmax())

            s2, r, term, trunc, _ = env.step(a)
            x2 = phi(s2).astype(np.float32)

            mem_s[cursor], mem_a[cursor], mem_r[cursor] = x, a, r
            mem_s2[cursor], mem_f[cursor] = x2, term
            cursor = (cursor + 1) % MEMORIA
            escritos = min(escritos + 1, MEMORIA)
            pasos += 1

            if escritos >= LOTE:
                i = rng.integers(0, escritos, size=LOTE)
                bs = torch.from_numpy(mem_s[i])
                ba = torch.from_numpy(mem_a[i])
                br = torch.from_numpy(mem_r[i])
                bs2 = torch.from_numpy(mem_s2[i])
                bf = torch.from_numpy(mem_f[i])

                with torch.no_grad():
                    mejor = objetivo(bs2).max(dim=1).values
                    y = br + GAMMA * mejor * (~bf)

                q = red(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                perdida = perdida_fn(q, y)

                opt.zero_grad()
                perdida.backward()          # <- el gradiente, en una linea
                opt.step()

            if pasos % REFRESCO == 0:
                objetivo.load_state_dict(red.state_dict())

            x = x2
            if term or trunc:
                break

    # ── evaluacion, con el mismo arnes de siempre ─────────────────────────
    X = torch.from_numpy(np.stack([phi(s) for s in range(env.n_states)]).astype(np.float32))
    with torch.no_grad():
        politica = red(X).argmax(dim=1).numpy()

    _, politica_optima, _ = value_iteration(rejilla(), gamma=GAMMA)
    opt_m, opt_lo, opt_hi = retorno_verdadero(rejilla(), politica_optima)
    real, lo, hi = retorno_verdadero(rejilla(), politica)

    print("\n  DQN en PyTorch · misma cuadricula, mismos hiperparametros\n")
    print(f"     retorno real            {real:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"     politica optima         {opt_m:+.4f}  [{opt_lo:+.4f}, {opt_hi:+.4f}]")
    print(f"     coincide con la optima  {int((politica == politica_optima).sum())} de {politica.size} casillas")
    print(f"     parametros de la red    {sum(p.numel() for p in red.parameters())}")
    print("\n  Compara con la version de NumPy:")
    print("     uv run python experiments/profundo.py --parte 1")
    print("\n  Los numeros no van a ser identicos y no tienen por que serlo. Lo que")
    print("  tiene que coincidir es la conclusion, dentro del intervalo.\n")


if __name__ == "__main__":
    main()
