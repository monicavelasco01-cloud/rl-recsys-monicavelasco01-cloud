"""Meridiano: el entorno del proyecto del modulo.

Un tutor que decide **que ejercicio poner a continuacion**. El agente elige una
habilidad de las 110 del catalogo, el estudiante la acierta o la falla, y esa
respuesta cambia lo que conviene poner despues.

═══════════════════════════════════════════════════════════════════════════
DE DONDE SALE EL ESTUDIANTE
═══════════════════════════════════════════════════════════════════════════

De un modelo ajustado sobre datos reales, no de una invencion. El conjunto es
ASSISTments 2009-2010: 4.151 secuencias de estudiantes, 325.637 interacciones y
110 habilidades con nombre.

El modelo es PFA (*Performance Factors Analysis*, Pavlik y otros, 2009). Para
cada habilidad ``k`` estima la probabilidad de acertar como

    P(acierto) = sigmoide( beta_k + gamma_k * aciertos_previos + rho_k * fallos_previos )

Tres numeros por habilidad. ``beta`` es la dificultad, ``gamma`` cuanto ensena
un acierto y ``rho`` cuanto ensena un fallo. Es el modelo de trazado de
conocimiento mas simple que se sostiene, y ajustado sobre estos datos acierta
el 69,2 % de las respuestas del conjunto de prueba frente al 65,9 % de predecir
siempre lo mas comun, con un AUC de 0,682.

═══════════════════════════════════════════════════════════════════════════
LO QUE ESTO ES Y LO QUE NO ES
═══════════════════════════════════════════════════════════════════════════

**Es** un simulador honesto: sus parametros salen de datos reales y su calidad
predictiva esta medida y publicada arriba.

**No es** un estudiante. Es un modelo de estudiante, y un agente entrenado aqui
esta optimizando contra ese modelo, no contra una persona. Cualquier conclusion
del proyecto lleva esa limitacion pegada, y decirla es parte de lo que se
califica. Es la misma disciplina de las sesiones 1 a 4, ahora sin la red de
seguridad de conocer la respuesta correcta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

RAIZ_DATOS = Path(__file__).resolve().parents[2] / "datos" / "assistments-2009"


# ══════════════════════════════════════════════════════════════════════════
# 1 · Los datos
# ══════════════════════════════════════════════════════════════════════════

def cargar_secuencias(ruta: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lee el formato de tres lineas por estudiante: largo, habilidades, aciertos."""
    lineas = Path(ruta).read_text().strip().split("\n")
    salida = []
    for i in range(len(lineas) // 3):
        habilidades = np.array([int(x) for x in lineas[i * 3 + 1].split(",")], dtype=int)
        aciertos = np.array([int(x) for x in lineas[i * 3 + 2].split(",")], dtype=int)
        salida.append((habilidades, aciertos))
    return salida


def nombres_de_habilidades(ruta: Path) -> dict[int, str]:
    nombres = {}
    for linea in Path(ruta).read_text().strip().split("\n"):
        if "\t" in linea:
            k, nombre = linea.split("\t", 1)
            nombres[int(k)] = nombre.strip()
    return nombres


# ══════════════════════════════════════════════════════════════════════════
# 2 · El modelo de estudiante
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ModeloPFA:
    """Tres numeros por habilidad, y lo que se puede hacer con ellos."""

    beta: np.ndarray      # dificultad
    gamma: np.ndarray     # cuanto ensena un acierto
    rho: np.ndarray       # cuanto ensena un fallo
    nombres: dict[int, str]

    @property
    def n_habilidades(self) -> int:
        return self.beta.size

    def probabilidad(self, habilidad: int, aciertos: float, fallos: float) -> float:
        z = self.beta[habilidad] + self.gamma[habilidad] * aciertos + self.rho[habilidad] * fallos
        return float(1.0 / (1.0 + np.exp(-z)))

    def probabilidades(self, aciertos: np.ndarray, fallos: np.ndarray) -> np.ndarray:
        """Probabilidad de acertar cada una de las habilidades, dado el historial."""
        z = self.beta + self.gamma * aciertos + self.rho * fallos
        return 1.0 / (1.0 + np.exp(-z))

    def guardar(self, ruta: Path) -> None:
        Path(ruta).write_text(json.dumps({
            "beta": self.beta.tolist(), "gamma": self.gamma.tolist(),
            "rho": self.rho.tolist(),
            "nombres": {str(k): v for k, v in self.nombres.items()},
        }))

    @classmethod
    def cargar(cls, ruta: Path) -> "ModeloPFA":
        d = json.loads(Path(ruta).read_text())
        return cls(np.array(d["beta"]), np.array(d["gamma"]), np.array(d["rho"]),
                   {int(k): v for k, v in d["nombres"].items()})


def _rasgos(seqs, n_habilidades: int):
    """Convierte las secuencias en (habilidad, aciertos previos, fallos previos, resultado)."""
    H, A, F, Y = [], [], [], []
    for habilidades, aciertos in seqs:
        ac = np.zeros(n_habilidades)
        fa = np.zeros(n_habilidades)
        for k, y in zip(habilidades, aciertos):
            H.append(k)
            A.append(ac[k])
            F.append(fa[k])
            Y.append(y)
            if y:
                ac[k] += 1.0
            else:
                fa[k] += 1.0
    return (np.array(H), np.array(A, dtype=float),
            np.array(F, dtype=float), np.array(Y, dtype=float))


def ajustar_pfa(raiz: Path = RAIZ_DATOS, pasos: int = 400, lr: float = 0.05) -> ModeloPFA:
    """Ajusta el modelo sobre el conjunto de entrenamiento. Tarda unos tres segundos.

    El descenso es por habilidad: cada una tiene sus tres parametros y su propio
    gradiente promediado sobre sus interacciones. No hay regularizacion, y con
    110 habilidades y 224.000 interacciones no hace falta.
    """
    crudo = Path(raiz) / "crudo"
    entrena = cargar_secuencias(crudo / "train.csv")
    prueba = cargar_secuencias(crudo / "test.csv")
    K = 1 + max(int(h.max()) for h, _ in entrena + prueba)

    H, A, F, Y = _rasgos(entrena, K)
    beta, gamma, rho = np.zeros(K), np.zeros(K), np.zeros(K)
    cuenta = np.maximum(np.bincount(H, minlength=K), 1)

    for _ in range(pasos):
        z = beta[H] + gamma[H] * A + rho[H] * F
        error = 1.0 / (1.0 + np.exp(-z)) - Y
        beta -= lr * np.bincount(H, weights=error, minlength=K) / cuenta
        gamma -= lr * np.bincount(H, weights=error * A, minlength=K) / cuenta
        rho -= lr * np.bincount(H, weights=error * F, minlength=K) / cuenta

    return ModeloPFA(beta, gamma, rho, nombres_de_habilidades(crudo / "skills.tsv"))


def calidad(modelo: ModeloPFA, raiz: Path = RAIZ_DATOS) -> dict[str, float]:
    """Acierto y AUC sobre el conjunto de prueba, y la tasa base para comparar.

    Sin esta funcion el simulador seria una caja negra en la que hay que creer.
    Con ella, cualquiera puede comprobar cuanto vale el modelo de estudiante
    antes de construir nada encima.
    """
    prueba = cargar_secuencias(Path(raiz) / "crudo" / "test.csv")
    H, A, F, Y = _rasgos(prueba, modelo.n_habilidades)
    p = 1.0 / (1.0 + np.exp(-(modelo.beta[H] + modelo.gamma[H] * A + modelo.rho[H] * F)))

    orden = np.argsort(p)
    y = Y[orden]
    rangos = np.arange(1, len(y) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    auc = float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    return {
        "acierto": float(((p > 0.5) == Y).mean()),
        "auc": auc,
        "tasa_base": float(max(Y.mean(), 1 - Y.mean())),
        "interacciones": int(len(Y)),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3 · El entorno
# ══════════════════════════════════════════════════════════════════════════

class Meridiano:
    """El tutor, como entorno con la misma interfaz que ``GridWorld``.

    Parameters
    ----------
    modelo:
        El modelo de estudiante ya ajustado.
    recompensa:
        ``"aciertos"`` da ``+1`` por cada respuesta correcta. Es la recompensa
        que sale sola, y **es la trampa de la sesion 3 otra vez**: el agente
        aprende a repetir la habilidad mas facil del catalogo.
        ``"dominio"`` da la mejora de dominio medio que produjo la interaccion.
        Premia ensenar, no acertar.
    n_pasos:
        Ejercicios por episodio.
    catalogo:
        Cuantas habilidades entran. Con ``None`` entran las 110. Reducirlo hace
        el problema manejable para una linea base tabular.

    Notes
    -----
    El estado que se devuelve es el par de vectores de aciertos y fallos por
    habilidad, aplanado. Con el catalogo completo son 220 numeros. **No es un
    estado pequeno y no cabe en una tabla**, que es exactamente el motivo por el
    que el bloque 2 existe.
    """

    def __init__(
        self,
        modelo: ModeloPFA,
        recompensa: Literal["aciertos", "dominio"] = "dominio",
        n_pasos: int = 50,
        catalogo: int | None = 20,
    ) -> None:
        if recompensa not in ("aciertos", "dominio"):
            raise ValueError(f"recompensa desconocida: {recompensa!r}")
        self.modelo = modelo
        self.recompensa = recompensa
        self.n_pasos = n_pasos

        # Las habilidades mas frecuentes primero: con un catalogo reducido
        # interesa quedarse con aquellas de las que hay datos suficientes.
        self.habilidades = np.arange(modelo.n_habilidades if catalogo is None else catalogo)
        self.n_actions = len(self.habilidades)
        self.n_caracteristicas = 2 * self.n_actions

        self._rng = np.random.default_rng()
        self._aciertos = np.zeros(self.n_actions)
        self._fallos = np.zeros(self.n_actions)
        self._t = 0

    # ── estado ─────────────────────────────────────────────────────────────
    def observacion(self) -> np.ndarray:
        return np.concatenate([self._aciertos, self._fallos])

    def dominio(self) -> float:
        """Probabilidad media de acertar sobre el catalogo. Es lo que se quiere subir."""
        p = self.modelo.probabilidades(
            np.pad(self._aciertos, (0, self.modelo.n_habilidades - self.n_actions)),
            np.pad(self._fallos, (0, self.modelo.n_habilidades - self.n_actions)),
        )
        return float(p[self.habilidades].mean())

    # ── interfaz ───────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._aciertos = np.zeros(self.n_actions)
        self._fallos = np.zeros(self.n_actions)
        self._t = 0
        return self.observacion(), {"dominio": self.dominio()}

    def step(self, accion: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        k = int(self.habilidades[accion])
        antes = self.dominio()
        p = self.modelo.probabilidad(k, self._aciertos[accion], self._fallos[accion])
        acierto = bool(self._rng.random() < p)

        if acierto:
            self._aciertos[accion] += 1.0
        else:
            self._fallos[accion] += 1.0
        self._t += 1

        despues = self.dominio()
        if self.recompensa == "aciertos":
            r = 1.0 if acierto else 0.0
        else:
            r = float(despues - antes) * 100.0   # escalado para que no sea diminuto

        terminado = self._t >= self.n_pasos
        return (self.observacion(), r, terminado, False,
                {"acierto": acierto, "habilidad": k, "dominio": despues, "p": p})


def medir(env: Meridiano, politica, episodios: int = 200, base_seed: int = 0) -> dict[str, float]:
    """Mide una politica con las TRES cifras que importan, no con una.

    - ``dominio_final``: cuanto sabe el estudiante al terminar. Es el objetivo.
    - ``aciertos``: cuantas respondio bien. Es lo que se ve, y engana.
    - ``habilidades_distintas``: cuantas habilidades distintas le puso. Es el
      detector de trampa: un agente que repite la mas facil saca muchos
      aciertos y ensena poco.
    """
    rng = np.random.default_rng(base_seed)
    dominios, aciertos, distintas, dominio_inicial = [], [], [], []
    for _ in range(episodios):
        obs, info = env.reset(seed=int(rng.integers(2**31)))
        dominio_inicial.append(info["dominio"])
        vistas, ok = set(), 0
        while True:
            a = politica(obs)
            obs, r, fin, _, info = env.step(a)
            vistas.add(info["habilidad"])
            ok += int(info["acierto"])
            if fin:
                break
        dominios.append(info["dominio"])
        aciertos.append(ok / env.n_pasos)
        distintas.append(len(vistas))
    return {
        "dominio_final": float(np.mean(dominios)),
        "dominio_inicial": float(np.mean(dominio_inicial)),
        "ganancia": float(np.mean(dominios) - np.mean(dominio_inicial)),
        "aciertos": float(np.mean(aciertos)),
        "habilidades_distintas": float(np.mean(distintas)),
    }
