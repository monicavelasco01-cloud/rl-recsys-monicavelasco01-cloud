"""Gradiente de politica: optimizar la conducta sin pasar por los valores.

Todo lo que llevamos hecho sigue el mismo esquema: estimar cuanto vale cada
accion y luego actuar avidamente sobre esa estimacion. La politica es una
consecuencia de los valores, no algo que se ajuste directamente.

Aqui se da la vuelta al planteamiento. La politica **es** los parametros: una
red que recibe el estado y devuelve una probabilidad para cada accion, y esos
parametros se mueven en la direccion que aumenta el retorno esperado. Ya no hay
un ``argmax`` en ninguna parte.

Eso resuelve de golpe dos cosas que con valores eran incomodas:

- Las acciones continuas. Sin lista sobre la que tomar el maximo, los metodos
  de valor no tienen por donde empezar. Es el problema del equipo B.
- Las politicas estocasticas. A veces la mejor conducta es genuinamente
  aleatoria, y una politica avida no puede representarla.

Y crea una nueva, que es el hilo de la sesion: **la varianza**. REINFORCE es
insesgado y ruidoso; casi todo lo que vino despues son maneras de bajarle el
ruido sin estropear el sesgo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aprox import Caracteristicas
from .envs import GridWorld
from .redes import Adam, Red


def softmax(z: np.ndarray) -> np.ndarray:
    """Convierte puntuaciones en probabilidades, restando el maximo por estabilidad.

    Sin la resta, ``exp`` de un numero grande desborda y salen ``nan``. Es una
    linea que parece cosmetica y no lo es: es la diferencia entre que esto
    entrene y que produzca basura silenciosamente.
    """
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class EntrenamientoPG:
    """Lo que devuelve un metodo de gradiente de politica."""

    red: Red
    retornos: np.ndarray
    nombre: str
    varianza_gradiente: float = 0.0

    def probabilidades(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.red.adelante(x))

    def politica(self, env: GridWorld, phi: Caracteristicas) -> np.ndarray:
        X = np.stack([phi(s) for s in range(env.n_states)])
        return softmax(self.red.adelante(X)).argmax(axis=1)

    def __str__(self) -> str:
        return (f"{self.nombre:<26s} retorno final "
                f"{float(np.mean(self.retornos[-100:])):+.4f}   "
                f"varianza del gradiente {self.varianza_gradiente:.4f}")


def _gradiente_entropia(p: np.ndarray) -> np.ndarray:
    """Gradiente de **menos** la entropia respecto a las puntuaciones.

    Se suma a la perdida para castigar las politicas que se vuelven
    deterministas antes de tiempo. Sin este termino, el gradiente de politica
    tiene un modo de fallo que no es raro y del que **no se recupera**: si la
    politica deja de llegar a la meta, todos los episodios salen igual de mal,
    el gradiente deja de contener informacion sobre como mejorar, y ahi se
    queda para siempre. Un metodo de valor con epsilon-avida no puede caer en
    eso, porque nunca deja de explorar.

    La cuenta, para quien la quiera seguir: con ``H = -sum p log p`` y ``p``
    saliendo de una softmax, resulta ``dH/dz_i = -p_i (log p_i + H)``. Aqui se
    devuelve el gradiente de ``-H``, que es el que hay que sumar a una perdida
    que se minimiza.
    """
    logp = np.log(np.clip(p, 1e-12, None))
    H = -(p * logp).sum(axis=1, keepdims=True)
    return p * (logp + H)


def _episodio(env: GridWorld, phi: Caracteristicas, red: Red,
              rng: np.random.Generator, max_pasos: int = 300):
    """Genera un episodio siguiendo la politica actual y devuelve su historia."""
    s, _ = env.reset(seed=int(rng.integers(2**31)))
    xs, acciones, recompensas = [], [], []
    for _ in range(max_pasos):
        x = phi(s)
        p = softmax(red.adelante(x)[0])
        a = int(rng.choice(len(p), p=p))
        s2, r, term, trunc, _ = env.step(a)
        xs.append(x)
        acciones.append(a)
        recompensas.append(r)
        s = s2
        if term or trunc:
            break
    return np.array(xs), np.array(acciones), np.array(recompensas)


def _retornos_por_paso(recompensas: np.ndarray, gamma: float) -> np.ndarray:
    """Retorno descontado desde cada paso hasta el final del episodio."""
    g = np.empty_like(recompensas, dtype=float)
    acumulado = 0.0
    for t in range(len(recompensas) - 1, -1, -1):
        acumulado = recompensas[t] + gamma * acumulado
        g[t] = acumulado
    return g


def reinforce(
    env: GridWorld,
    phi: Caracteristicas,
    episodes: int = 1500,
    gamma: float = 0.99,
    lr: float = 5e-3,
    oculta: int = 64,
    linea_base: bool = False,
    entropia: float = 0.0,
    episodios_por_lote: int = 1,
    seed: int = 0,
    nombre: str | None = None,
) -> EntrenamientoPG:
    """REINFORCE, con o sin linea base.

    La regla, en una linea: **sube la probabilidad de lo que hiciste, en
    proporcion a lo bien que salio**. Si el retorno fue alto, la accion que se
    tomo se vuelve mas probable; si fue bajo, menos.

    Parameters
    ----------
    linea_base:
        ``False`` usa el retorno tal cual. ``True`` le resta una media movil de
        los retornos de los episodios **anteriores**. Restar una cantidad que no
        depende del episodio actual no cambia la direccion esperada del
        gradiente, y sin embargo baja su varianza.

        El detalle de que sea de los episodios anteriores no es un capricho: si
        se resta la media del propio episodio, la linea base queda
        correlacionada con los retornos que multiplica y el estimador **se
        sesga**. Esta implementado de las dos maneras en el laboratorio,
        precisamente para que se mida la diferencia.

    Notes
    -----
    El gradiente que se implementa es el de ``-log pi(a|s) * ventaja``, que es
    la version que hay que minimizar. Para la softmax, la derivada respecto a
    las puntuaciones tiene una forma muy simple: ``p - e_a``, donde ``e_a`` es
    el vector con un uno en la accion tomada. Esa simplicidad es la razon por
    la que la softmax es la eleccion estandar aqui.
    """
    rng = np.random.default_rng(seed)
    d = phi(0).size
    red = Red((d, oculta, env.n_actions), seed=seed)
    opt = Adam(red.parametros(), lr=lr)
    retornos = np.empty(episodes, dtype=float)
    normas_grad = []
    base = 0.0          # media movil de los retornos ya vistos
    vistos = 0
    ep = 0

    while ep < episodes:
        # Se juntan varios episodios y se da UN paso con todos. Actualizar con
        # un solo episodio es lo que dice la formula y es inutilizable en la
        # practica: el gradiente de un episodio es tan ruidoso que la politica
        # se desploma. El laboratorio lo mide bajando este numero a 1.
        XS, AC, VE = [], [], []
        for _ in range(min(episodios_por_lote, episodes - ep)):
            xs, acciones, recompensas = _episodio(env, phi, red, rng)
            retornos[ep] = float(recompensas.sum())
            ep += 1
            g = _retornos_por_paso(recompensas, gamma)
            VE.append(g - base if linea_base else g)
            XS.append(xs)
            AC.append(acciones)
            vistos += 1
            base += (float(g.mean()) - base) / vistos

        xs = np.concatenate(XS)
        acciones = np.concatenate(AC)
        ventaja = np.concatenate(VE)

        # Se promedia sobre todos los pasos del lote. Es una de las dos
        # normalizaciones razonables y es la que midio estable aqui; la otra,
        # promediar por episodio, se prueba en el laboratorio.
        p = softmax(red.adelante(xs))
        grad = p.copy()
        grad[np.arange(len(acciones)), acciones] -= 1.0
        grad *= ventaja[:, None] / len(acciones)
        grad += entropia * _gradiente_entropia(p) / len(acciones)

        red.atras(grad)
        normas_grad.append(float(np.sqrt(sum(float((gr ** 2).sum())
                                             for gr in red.gradientes() if gr is not None))))
        opt.paso(red.parametros(), red.gradientes())

    if nombre is None:
        nombre = "REINFORCE con línea base" if linea_base else "REINFORCE"
    return EntrenamientoPG(red=red, retornos=retornos, nombre=nombre,
                           varianza_gradiente=float(np.var(normas_grad)) if normas_grad else 0.0)


def actor_critico(
    env: GridWorld,
    phi: Caracteristicas,
    episodes: int = 1500,
    gamma: float = 0.99,
    lr_actor: float = 5e-3,
    lr_critico: float = 1e-2,
    oculta: int = 64,
    entropia: float = 0.0,
    episodios_por_lote: int = 1,
    seed: int = 0,
) -> EntrenamientoPG:
    """Actor-critico: la linea base deja de ser una constante y pasa a aprenderse.

    En REINFORCE con linea base restabamos la media del episodio, que es la
    misma para todos los pasos. Aqui se resta ``V(s)``, que depende del estado:
    una accion se premia si salio mejor **de lo que cabia esperar desde ahi**,
    no mejor que la media global.

    Son dos redes. El **actor** decide y el **critico** juzga. El critico se
    entrena como cualquier estimador de valor, con el mismo error temporal de la
    sesion 2. Es el punto donde las dos mitades del curso se juntan.
    """
    rng = np.random.default_rng(seed)
    d = phi(0).size
    actor = Red((d, oculta, env.n_actions), seed=seed)
    critico = Red((d, oculta, 1), seed=seed + 1000)
    opt_a = Adam(actor.parametros(), lr=lr_actor)
    opt_c = Adam(critico.parametros(), lr=lr_critico)
    retornos = np.empty(episodes, dtype=float)
    ep = 0

    while ep < episodes:
        XS, AC, GG = [], [], []
        for _ in range(min(episodios_por_lote, episodes - ep)):
            xs_i, ac_i, rec_i = _episodio(env, phi, actor, rng)
            retornos[ep] = float(rec_i.sum())
            ep += 1
            XS.append(xs_i)
            AC.append(ac_i)
            GG.append(_retornos_por_paso(rec_i, gamma))
        xs = np.concatenate(XS)
        acciones = np.concatenate(AC)
        g = np.concatenate(GG)

        valores = critico.adelante(xs)[:, 0]
        ventaja = g - valores

        # crítico: acercar V(s) al retorno observado
        critico.atras((2.0 * (valores - g) / len(g))[:, None])
        opt_c.paso(critico.parametros(), critico.gradientes())

        # actor: mismo gradiente que REINFORCE, con la ventaja del crítico
        p = softmax(actor.adelante(xs))
        grad = p.copy()
        grad[np.arange(len(acciones)), acciones] -= 1.0
        grad *= ventaja[:, None] / len(acciones)
        grad += entropia * _gradiente_entropia(p) / len(acciones)
        actor.atras(grad)
        opt_a.paso(actor.parametros(), actor.gradientes())

    return EntrenamientoPG(red=actor, retornos=retornos, nombre="actor-crítico")
