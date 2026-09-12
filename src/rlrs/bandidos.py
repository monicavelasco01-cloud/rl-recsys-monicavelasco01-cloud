"""Bandidos: la decision repetida, sin estado.

El bloque 2 termino con un agente que aprende una politica sobre un estado que
no cabe en una tabla. Este bloque quita el estado y se queda con lo minimo: hay
que elegir una opcion entre varias, una y otra vez, y solo se ve el resultado de
la que se eligio.

Es menos que un MDP, y por eso importa: **un recomendador es un bandido antes
que un MDP**. Cuando se sirve una tanda de articulos no se sabe que habria
pasado con los otros, y la respuesta del usuario llega enseguida en vez de
cincuenta pasos despues.

Tres cosas viven aqui:

- ``BanditoBernoulli``: el bandido de libro, con brazos de probabilidad fija.
  Sirve para ver el arrepentimiento crecer y comparar cotas.
- Las politicas: ``avida``, ``epsilon_avida``, ``ucb1`` y ``thompson``. Todas
  tienen la misma firma y devuelven un ``Corrida``.
- ``MeridianoBandido``: la vista de bandido sobre el tutor del proyecto. Es
  **deliberadamente incorrecta**, y de esa incorreccion trata la sesion: tirar
  de un brazo cambia la distribucion de ese brazo, porque el estudiante
  aprende. Un bandido estacionario supone justo lo contrario.

Nada de este modulo usa el estado. Esa es toda la idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .meridiano import Meridiano


# ── el bandido de libro ──────────────────────────────────────────────────────
@dataclass
class BanditoBernoulli:
    """Brazos con probabilidad de exito fija y conocida por el entorno.

    Que las probabilidades sean fijas es la hipotesis de estacionariedad, y es
    lo que permite hablar de «el mejor brazo» en singular. En cuanto tirar de un
    brazo lo cambia, la frase deja de tener sentido.
    """

    p: np.ndarray

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=float)
        if self.p.ndim != 1 or not np.all((self.p >= 0) & (self.p <= 1)):
            raise ValueError("p tiene que ser un vector de probabilidades")

    @property
    def n_brazos(self) -> int:
        return int(self.p.size)

    @property
    def mejor(self) -> int:
        return int(self.p.argmax())

    @property
    def p_mejor(self) -> float:
        return float(self.p.max())

    def tirar(self, brazo: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self.p[brazo])


def bandido_dificil(n_brazos: int = 20, brecha: float = 0.05, seed: int = 0) -> BanditoBernoulli:
    """Un bandido donde el mejor brazo esta poco separado del segundo.

    La dificultad de un bandido no la da el numero de brazos: la da la brecha
    entre el mejor y los demas. Con brechas grandes cualquier politica acierta
    enseguida y no hay nada que ensenar.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.30, 0.50, size=n_brazos)
    p[rng.integers(n_brazos)] = p.max() + brecha
    return BanditoBernoulli(np.clip(p, 0.0, 1.0))


# ── lo que devuelve una corrida ──────────────────────────────────────────────
@dataclass
class Corrida:
    """El historial de una politica sobre un bandido."""

    nombre: str
    brazos: np.ndarray = field(repr=False)      # que brazo se tiro en cada ronda
    recompensas: np.ndarray = field(repr=False)
    p_mejor: float

    @property
    def rondas(self) -> int:
        return int(self.brazos.size)

    @property
    def recompensa_total(self) -> float:
        return float(self.recompensas.sum())

    def arrepentimiento(self, p: np.ndarray) -> float:
        """Lo que se dejo de ganar frente a haber tirado siempre del mejor.

        Se calcula con las probabilidades VERDADERAS, no con lo observado. Esa
        es la trampa del arrepentimiento: solo se puede medir si se conoce la
        respuesta, y por eso fuera del laboratorio no se puede medir.
        """
        return float(self.rondas * self.p_mejor - np.asarray(p)[self.brazos].sum())

    def curva_arrepentimiento(self, p: np.ndarray) -> np.ndarray:
        return np.cumsum(self.p_mejor - np.asarray(p)[self.brazos])

    def reparto(self, n_brazos: int) -> np.ndarray:
        return np.bincount(self.brazos, minlength=n_brazos)


# ── las politicas ────────────────────────────────────────────────────────────
def _corrida(nombre, brazos, recompensas, p_mejor) -> Corrida:
    return Corrida(nombre=nombre,
                   brazos=np.asarray(brazos, dtype=int),
                   recompensas=np.asarray(recompensas, dtype=float),
                   p_mejor=p_mejor)


def epsilon_avida(bandido, rondas: int, epsilon: float = 0.1, seed: int = 0) -> Corrida:
    """Con probabilidad epsilon prueba al azar; el resto del tiempo explota.

    Con ``epsilon = 0`` es la avida pura: se queda con el primer brazo que le
    parezca bueno y no vuelve a mirar. Es la politica que todo el mundo
    descarta en la primera clase, y conviene medirla antes de descartarla.
    """
    rng = np.random.default_rng(seed)
    k = bandido.n_brazos
    sumas = np.zeros(k)
    tiradas = np.zeros(k)
    brazos, recompensas = [], []
    for _ in range(rondas):
        if rng.random() < epsilon or tiradas.sum() == 0:
            a = int(rng.integers(k))
        else:
            medias = np.divide(sumas, tiradas, out=np.zeros(k), where=tiradas > 0)
            # Un brazo sin tirar cuenta como media cero, no como infinito: eso
            # es lo que hace que la avida pura se quede pegada a lo primero
            # que le funciono, que es justo lo que queremos poder medir.
            a = int(rng.choice(np.flatnonzero(medias == medias.max())))
        r = bandido.tirar(a, rng)
        sumas[a] += r
        tiradas[a] += 1
        brazos.append(a)
        recompensas.append(r)
    nombre = "ávida pura" if epsilon == 0 else f"ε-ávida (ε={epsilon:g})"
    return _corrida(nombre, brazos, recompensas, bandido.p_mejor)


def avida(bandido, rondas: int, seed: int = 0) -> Corrida:
    return epsilon_avida(bandido, rondas, epsilon=0.0, seed=seed)


def ucb1(bandido, rondas: int, c: float = 2.0, seed: int = 0) -> Corrida:
    """Optimismo ante la incertidumbre: se premia al brazo poco visitado.

    Antes de nada tira una vez de cada brazo. Con veinte brazos, eso son veinte
    rondas gastadas antes de empezar a decidir, y en un horizonte corto veinte
    rondas son mucho. Ese detalle no suele aparecer en las cotas.
    """
    rng = np.random.default_rng(seed)
    k = bandido.n_brazos
    sumas = np.zeros(k)
    tiradas = np.zeros(k)
    brazos, recompensas = [], []
    for t in range(rondas):
        sin_tirar = np.flatnonzero(tiradas == 0)
        if sin_tirar.size:
            a = int(sin_tirar[0])
        else:
            medias = sumas / tiradas
            cota = medias + np.sqrt(c * np.log(t + 1) / tiradas)
            a = int(rng.choice(np.flatnonzero(cota == cota.max())))
        r = bandido.tirar(a, rng)
        sumas[a] += r
        tiradas[a] += 1
        brazos.append(a)
        recompensas.append(r)
    return _corrida(f"UCB1 (c={c:g})", brazos, recompensas, bandido.p_mejor)


def thompson(bandido, rondas: int, seed: int = 0) -> Corrida:
    """Muestrea una creencia sobre cada brazo y juega la mejor muestra.

    Con recompensas de cero o uno, la creencia es una Beta y la actualizacion
    es contar exitos y fracasos. Son cuatro lineas y suele ganar.
    """
    rng = np.random.default_rng(seed)
    k = bandido.n_brazos
    exitos = np.ones(k)
    fracasos = np.ones(k)
    brazos, recompensas = [], []
    for _ in range(rondas):
        muestra = rng.beta(exitos, fracasos)
        a = int(muestra.argmax())
        r = bandido.tirar(a, rng)
        if not (0.0 <= r <= 1.0):
            raise ValueError(
                f"Thompson con creencia Beta exige recompensas entre 0 y 1, y este "
                f"brazo devolvió {r:.4f}.\n"
                "No es un fallo de implementación: la Beta es la creencia conjugada de "
                "una Bernoulli, así que este Thompson solo sirve donde la recompensa es "
                "acertar o fallar.\n"
                "En Meridiano eso significa recompensa='aciertos'. Con "
                "recompensa='dominio' hace falta otra creencia, y elegirla es una "
                "decisión de diseño, no un detalle."
            )
        exitos[a] += r
        fracasos[a] += 1.0 - r
        brazos.append(a)
        recompensas.append(r)
    return _corrida("Thompson", brazos, recompensas, bandido.p_mejor)


def linucb(bandido, rondas: int, alpha: float = 1.0, seed: int = 0,
           contexto=None) -> Corrida:
    """Bandido contextual lineal: la recompensa depende de lo que se observa.

    Es el primer metodo del bloque que vuelve a mirar el estado, y por eso es la
    bisagra. Supone que la recompensa esperada de un brazo es ``theta_a . x``,
    con ``x`` el contexto de esta ronda, y mantiene por brazo una regresion
    lineal con su incertidumbre.

    ``alpha`` es el mismo optimismo de UCB1, aqui sobre la elipse de confianza
    de la regresion. Con ``alpha = 0`` es regresion codiciosa y sin exploracion.

    Si ``contexto`` es ``None`` se usa un contexto constante de un solo numero.
    Ojo con la intuicion facil: eso **no** lo convierte en UCB1. La estimacion
    pasa a ser una regresion regularizada sobre la media y el radio de confianza
    sale de la elipse, no de la raiz de log t partido por n. Medido sobre el
    mismo bandido, 2000 rondas y diez semillas, LinUCB con contexto constante da
    96,87 de arrepentimiento y UCB1 con c=1 da 188,72. Se parecen en la idea y
    no en el numero.
    """
    rng = np.random.default_rng(seed)
    k = bandido.n_brazos
    if contexto is None:
        contexto = lambda: np.ones(1)
    d = int(np.asarray(contexto()).size)

    A = np.stack([np.eye(d) for _ in range(k)])       # matrices de covarianza
    b_vec = np.zeros((k, d))
    brazos, recompensas = [], []
    for _ in range(rondas):
        x = np.asarray(contexto(), dtype=float).reshape(d)
        puntuacion = np.empty(k)
        for a in range(k):
            A_inv = np.linalg.inv(A[a])
            theta = A_inv @ b_vec[a]
            puntuacion[a] = theta @ x + alpha * np.sqrt(x @ A_inv @ x)
        a = int(rng.choice(np.flatnonzero(puntuacion == puntuacion.max())))
        r = bandido.tirar(a, rng)
        A[a] += np.outer(x, x)
        b_vec[a] += r * x
        brazos.append(a)
        recompensas.append(r)
    return _corrida(f"LinUCB (α={alpha:g})", brazos, recompensas, bandido.p_mejor)


POLITICAS = {
    "ávida pura": lambda b, n, s: epsilon_avida(b, n, epsilon=0.0, seed=s),
    "ε-ávida 0,10": lambda b, n, s: epsilon_avida(b, n, epsilon=0.10, seed=s),
    "UCB1": lambda b, n, s: ucb1(b, n, seed=s),
    "Thompson": lambda b, n, s: thompson(b, n, seed=s),
}


def comparar(bandido, rondas: int, semillas: int = 20) -> dict[str, dict[str, float]]:
    """Cada politica sobre el mismo bandido, con varias semillas e intervalo."""
    salida = {}
    for nombre, hacer in POLITICAS.items():
        arr = np.array([hacer(bandido, rondas, s).arrepentimiento(bandido.p)
                        for s in range(semillas)], dtype=float)
        h = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        salida[nombre] = {"media": float(arr.mean()),
                          "lo": float(arr.mean() - h),
                          "hi": float(arr.mean() + h)}
    return salida


def barrido_horizonte(bandido, horizontes, semillas: int = 20) -> dict[int, dict]:
    """El mismo experimento a varios horizontes. Es el reto de la sesion."""
    return {int(n): comparar(bandido, int(n), semillas) for n in horizontes}


# ── la vista de bandido sobre Meridiano ──────────────────────────────────────
class MeridianoBandido:
    """El tutor visto como bandido: un brazo por habilidad, sin estado.

    **Esta vista es incorrecta a proposito.** Un bandido estacionario supone que
    tirar de un brazo no cambia su distribucion, y aqui la cambia: cada vez que
    se pone una habilidad, el estudiante la practica y la probabilidad de
    acertarla sube. El brazo se agota.

    Se incluye para poder medir hasta donde llega la aproximacion y donde se
    rompe, que es la pregunta de la sesion: **cuando un recomendador deja de
    ser un bandido**.
    """

    def __init__(self, modelo, catalogo: int = 20, rondas: int = 500,
                 recompensa: str = "dominio") -> None:
        self.env = Meridiano(modelo, recompensa=recompensa,
                             n_pasos=rondas, catalogo=catalogo)
        self.rondas = rondas

    @property
    def n_brazos(self) -> int:
        return self.env.n_actions

    def reset(self, seed: int | None = None) -> None:
        self.env.reset(seed=seed)

    def tirar(self, brazo: int, rng: np.random.Generator) -> float:
        _, r, _, _, _ = self.env.step(brazo)
        return float(r)

    # Sin brazo mejor fijo: por eso no hay `p_mejor` ni arrepentimiento.
    p_mejor = float("nan")


def jugar_meridiano(modelo, politica, rondas: int, seed: int = 0,
                    catalogo: int = 20, **kw) -> tuple[float, np.ndarray, float]:
    """Corre una politica de bandido sobre Meridiano y devuelve lo que importa.

    Returns
    -------
    ganancia:
        Cuanto subio el dominio medio entre el principio y el final.
    reparto:
        Cuantas veces se tiro de cada brazo.
    recompensa:
        La suma de recompensas, que es lo que la politica creia maximizar.
    """
    b = MeridianoBandido(modelo, catalogo=catalogo, rondas=rondas)
    b.reset(seed=seed)
    antes = b.env.dominio()
    corrida = politica(b, rondas, seed)
    despues = b.env.dominio()
    return float(despues - antes), corrida.reparto(b.n_brazos), corrida.recompensa_total
