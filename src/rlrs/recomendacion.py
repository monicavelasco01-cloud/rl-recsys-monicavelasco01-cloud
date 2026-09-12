"""Recomendacion: los datos, el juez y las lineas base.

El bloque 3 empezo quitando el estado y quedandose con la decision repetida.
Aqui se quita tambien la interaccion: hay un historial de lo que cada usuario
hizo, y hay que proponer lo siguiente. Es el problema donde de verdad se usa
todo esto.

Este modulo NO trae un algoritmo estrella. Trae **el juez**, que es lo que hay
que construir antes que cualquier algoritmo:

- ``particion_temporal`` y ``particion_aleatoria``, que son la misma idea con
  una diferencia que lo cambia todo.
- ``recall_en_k``, ``ndcg_en_k``, ``cobertura`` y ``novedad``, las cuatro
  cifras que hay que reportar juntas.
- Cuatro lineas base, dos de ellas humillantes.
- Un vecino mas cercano por items y una factorizacion implicita, para tener
  contra que comparar.

Los datos son los mismos de Meridiano: ASSISTments 2009-2010. Un usuario es un
estudiante, un item es una habilidad, y una interaccion es que la practico.
**La realimentacion es implicita**, y viene con un sesgo que no es un detalle:
que habilidad practico cada estudiante lo eligio el tutor de entonces, no el
estudiante. Los datos no dicen que le gusta a nadie: dicen que le pusieron
delante.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .meridiano import RAIZ_DATOS, cargar_secuencias, nombres_de_habilidades

MINIMO_INTERACCIONES = 10


# ══════════════════════════════════════════════════════════════════════════
# 1 · los datos
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Datos:
    """Las secuencias de practica, listas para partir.

    Attributes
    ----------
    secuencias:
        Una por usuario, con los items en el orden en que ocurrieron. El orden
        importa y es lo unico que permite partir por tiempo.
    n_items:
        Cuantos items hay en el catalogo.
    nombres:
        El nombre de cada item, para poder mirar una recomendacion y opinar.
    """

    secuencias: list[np.ndarray]
    n_items: int
    nombres: dict[int, str] = field(default_factory=dict)

    @property
    def n_usuarios(self) -> int:
        return len(self.secuencias)

    @property
    def n_interacciones(self) -> int:
        return int(sum(len(s) for s in self.secuencias))

    def popularidad(self) -> np.ndarray:
        """Cuantas veces se practico cada item, sobre TODO el conjunto.

        Ojo: esta se usa solo para describir los datos. La popularidad que
        alimenta la linea base se calcula sobre el entrenamiento y nada mas,
        porque si no seria fuga.
        """
        cuenta = np.zeros(self.n_items)
        for s in self.secuencias:
            np.add.at(cuenta, s, 1)
        return cuenta


def cargar(raiz: Path = RAIZ_DATOS, minimo: int = MINIMO_INTERACCIONES) -> Datos:
    """Lee ASSISTments y se queda con los usuarios que tienen historial suficiente.

    Un usuario con tres interacciones no se puede partir en entrenamiento y
    prueba de forma sensata, y ademas mete ruido en las medias. Descartarlos es
    una decision de diseno, no una limpieza inocente: **cambia el resultado**, y
    por eso el umbral es un parametro y no un numero escondido.
    """
    crudo = Path(raiz) / "crudo"
    seqs = [h for h, _ in cargar_secuencias(crudo / "train.csv")
            + cargar_secuencias(crudo / "test.csv")]
    n_items = 1 + max(int(s.max()) for s in seqs)
    return Datos(secuencias=[s for s in seqs if len(s) >= minimo],
                 n_items=n_items,
                 nombres=nombres_de_habilidades(crudo / "skills.tsv"))


# ══════════════════════════════════════════════════════════════════════════
# 2 · las particiones
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Particion:
    """Lo que ve el modelo y lo que se le esconde."""

    nombre: str
    train: list[np.ndarray]
    test: list[np.ndarray]
    n_items: int
    # Con quien se AJUSTA el modelo. Normalmente es `train`, y entonces vale
    # None. Solo cambia en arranque en frio, donde el modelo se ajusta con unos
    # usuarios y se evalua con otros a los que no ha visto nunca.
    ajuste: list[np.ndarray] | None = None

    @property
    def usuarios_evaluables(self) -> int:
        return sum(1 for t in self.test if len(t) > 0)

    @property
    def para_ajustar(self) -> list[np.ndarray]:
        """Los historiales con los que se ajusta. Todos los modelos usan esto."""
        return self.train if self.ajuste is None else self.ajuste


def particion_temporal(datos: Datos, fraccion: float = 0.8) -> Particion:
    """Se esconde **el final** de cada historial.

    Es la unica particion que responde a la pregunta que de verdad importa:
    dado lo que este usuario hizo hasta hoy, que va a hacer manana. Cualquier
    sistema en produccion se enfrenta a eso y no a otra cosa.
    """
    train, test = [], []
    for s in datos.secuencias:
        corte = max(1, int(len(s) * fraccion))
        train.append(s[:corte])
        test.append(s[corte:])
    return Particion("temporal", train, test, datos.n_items)


def particion_aleatoria(datos: Datos, fraccion: float = 0.8, seed: int = 0) -> Particion:
    """Se esconde un puñado de interacciones **al azar** de cada historial.

    Es la particion que sale por defecto en casi todos los tutoriales, y es la
    que hay que saber reconocer. El modelo ve el futuro del usuario mientras
    predice su pasado, asi que las cifras suben sin que nada haya mejorado.
    """
    rng = np.random.default_rng(seed)
    train, test = [], []
    for s in datos.secuencias:
        n = len(s)
        corte = max(1, int(n * fraccion))
        idx = rng.permutation(n)
        train.append(s[np.sort(idx[:corte])])
        test.append(s[np.sort(idx[corte:])])
    return Particion("aleatoria", train, test, datos.n_items)


# ══════════════════════════════════════════════════════════════════════════
# 3 · el juez
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Evaluacion:
    """Las cuatro cifras, juntas. Nunca una sola."""

    nombre: str
    recall: float
    ndcg: float
    cobertura: float          # fraccion del catalogo que llega a recomendarse
    novedad: float            # sorpresa media, en bits
    k: int
    n_usuarios: int
    # Fraccion de usuarios a los que el recomendador no supo que decirles y
    # devolvio una lista vacia. Normalmente es 0. Deja de serlo en arranque en
    # frio, y entonces es la cifra que mas dice de todas.
    sin_respuesta: float = 0.0

    def __str__(self) -> str:
        return (f"{self.nombre:<26s} Recall@{self.k} {self.recall:.4f}   "
                f"nDCG@{self.k} {self.ndcg:.4f}   cobertura {self.cobertura:.3f}   "
                f"novedad {self.novedad:.2f}")


def recall_en_k(recomendados: list[int], relevantes: set[int], k: int) -> float:
    """Que fraccion de lo que el usuario iba a hacer aparece en el top k.

    Se divide entre ``min(len(relevantes), k)`` y no entre ``len(relevantes)``:
    si un usuario tiene treinta items relevantes y solo se recomiendan diez, el
    maximo alcanzable es diez, y castigar por lo inalcanzable mide el tamano
    del historial en vez de la calidad del modelo.
    """
    if not relevantes:
        return 0.0
    aciertos = sum(1 for i in recomendados[:k] if i in relevantes)
    return aciertos / min(len(relevantes), k)


def ndcg_en_k(recomendados: list[int], relevantes: set[int], k: int) -> float:
    """Como el recall, pero premiando que los aciertos vayan arriba."""
    if not relevantes:
        return 0.0
    dcg = sum(1.0 / np.log2(pos + 2)
              for pos, item in enumerate(recomendados[:k]) if item in relevantes)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevantes), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def evaluar(particion: Particion, recomendador, k: int = 10,
            nombre: str | None = None) -> Evaluacion:
    """Corre un recomendador sobre una particion y devuelve las cuatro cifras.

    ``recomendador`` recibe el historial de entrenamiento de un usuario y
    devuelve una lista de items ordenada. No recibe el conjunto de prueba, y
    esa es toda la disciplina que hace falta para no hacer trampa sin querer.
    """
    recalls, ndcgs = [], []
    vacias = 0
    vistos: set[int] = set()
    # La novedad se mide con la popularidad del ENTRENAMIENTO: recomendar algo
    # raro es sorprendente solo respecto a lo que ya se sabia.
    cuenta = np.zeros(particion.n_items)
    for tr in particion.para_ajustar:
        np.add.at(cuenta, tr, 1)
    total = max(cuenta.sum(), 1.0)
    p_item = np.clip(cuenta / total, 1e-12, None)
    sorpresas = []

    for tr, te in zip(particion.train, particion.test):
        relevantes = set(int(x) for x in te)
        if not relevantes:
            continue
        top = [int(i) for i in recomendador(tr)][:k]
        if not top:
            vacias += 1
        vistos.update(top)
        recalls.append(recall_en_k(top, relevantes, k))
        ndcgs.append(ndcg_en_k(top, relevantes, k))
        if top:
            sorpresas.append(float(np.mean(-np.log2(p_item[top]))))

    return Evaluacion(
        nombre=nombre or getattr(recomendador, "nombre", "sin nombre"),
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        cobertura=len(vistos) / particion.n_items,
        novedad=float(np.mean(sorpresas)) if sorpresas else 0.0,
        k=k,
        n_usuarios=len(recalls),
        sin_respuesta=vacias / len(recalls) if recalls else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# 4 · las lineas base
# ══════════════════════════════════════════════════════════════════════════

def _con_nombre(f, nombre: str):
    f.nombre = nombre
    return f


def por_popularidad(particion: Particion, k: int = 50):
    """Lo mas practicado por todo el mundo, igual para todos.

    Es la linea base que hay que batir en cualquier articulo de recomendacion,
    y la que mas veces gana. No mira al usuario: por eso su cobertura es
    ridicula y conviene mirarla.
    """
    cuenta = np.zeros(particion.n_items)
    for tr in particion.para_ajustar:
        np.add.at(cuenta, tr, 1)
    orden = [int(i) for i in np.argsort(cuenta)[::-1]]
    return _con_nombre(lambda historial: orden[:k], "popularidad")


def al_azar(particion: Particion, seed: int = 0, k: int = 50):
    """Items al azar. Existe para que la cobertura tenga un techo con el que comparar."""
    rng = np.random.default_rng(seed)
    n = particion.n_items
    return _con_nombre(lambda historial: [int(i) for i in rng.permutation(n)[:k]], "al azar")


def repetir_lo_propio(particion: Particion, k: int = 50):
    """Lo que ese usuario ya practico, lo mas repetido primero, y despues popularidad.

    Dos lineas de codigo. Es la linea base que casi nadie reporta y la que
    casi siempre gana en realimentacion implicita, porque la gente repite. Si
    su modelo no le gana a esto, su modelo no ha aprendido nada sobre gustos:
    ha aprendido a contar.
    """
    cuenta = np.zeros(particion.n_items)
    for tr in particion.para_ajustar:
        np.add.at(cuenta, tr, 1)
    respaldo = [int(i) for i in np.argsort(cuenta)[::-1]]

    def recomendar(historial):
        propios = np.bincount(historial, minlength=particion.n_items)
        mios = [int(i) for i in np.argsort(propios)[::-1] if propios[i] > 0]
        resto = [i for i in respaldo if i not in set(mios)]
        return (mios + resto)[:k]

    return _con_nombre(recomendar, "repetir lo propio")


def novedad_pura(particion: Particion, k: int = 50):
    """Lo que ese usuario NO ha practicado, empezando por lo mas popular.

    Es la version del recomendador que muchos dan por supuesta: no repetir. Se
    incluye para poder medir cuanto cuesta esa suposicion, que en estos datos
    es mucho.
    """
    cuenta = np.zeros(particion.n_items)
    for tr in particion.para_ajustar:
        np.add.at(cuenta, tr, 1)
    orden = [int(i) for i in np.argsort(cuenta)[::-1]]

    def recomendar(historial):
        suyos = set(int(x) for x in historial)
        return [i for i in orden if i not in suyos][:k]

    return _con_nombre(recomendar, "novedad pura")


# ══════════════════════════════════════════════════════════════════════════
# 5 · dos modelos de verdad
# ══════════════════════════════════════════════════════════════════════════

def knn_items(particion: Particion, vecinos: int = 20, k: int = 50, repetir: bool = True):
    """Filtro colaborativo por co-ocurrencia de items.

    Dos items se parecen si los mismos usuarios los practicaron. La semejanza
    es el coseno sobre la matriz binaria usuario por item, que es la version
    mas simple que funciona.

    ``repetir`` decide si se permite recomendar algo que el usuario ya hizo.
    En estos datos eso cambia el resultado por completo, y por eso es un
    interruptor y no una decision escondida.
    """
    n = particion.n_items
    R = np.zeros((len(particion.para_ajustar), n), dtype=float)
    for u, tr in enumerate(particion.para_ajustar):
        R[u, np.unique(tr)] = 1.0

    normas = np.linalg.norm(R, axis=0)
    normas[normas == 0] = 1.0
    S = (R.T @ R) / np.outer(normas, normas)
    np.fill_diagonal(S, 0.0)

    # Solo los `vecinos` mas parecidos de cada item: el resto es ruido y
    # ademas ensucia la cobertura.
    if vecinos < n:
        umbral = np.partition(S, -vecinos, axis=1)[:, -vecinos][:, None]
        S = np.where(S >= umbral, S, 0.0)

    cuenta = np.zeros(n)
    for tr in particion.para_ajustar:
        np.add.at(cuenta, tr, 1)
    respaldo = [int(i) for i in np.argsort(cuenta)[::-1]]

    def recomendar(historial):
        perfil = np.zeros(n)
        np.add.at(perfil, historial, 1.0)
        puntos = S.T @ perfil
        if not repetir:
            puntos[np.unique(historial)] = -np.inf
        orden = [int(i) for i in np.argsort(puntos)[::-1] if np.isfinite(puntos[i])]
        vistos = set(orden[:k])
        return (orden[:k] + [i for i in respaldo if i not in vistos])[:k]

    return _con_nombre(recomendar, f"kNN por ítems ({vecinos})")


def factorizacion_implicita(particion: Particion, factores: int = 32, pasos: int = 15,
                            regular: float = 0.1, alfa: float = 20.0,
                            k: int = 50, seed: int = 0, repetir: bool = True):
    """Minimos cuadrados alternados para realimentacion implicita.

    Es el metodo de Hu, Koren y Volinsky (2008). La idea que lo distingue: en
    realimentacion implicita **no hay ceros negativos**. Que alguien no haya
    practicado una habilidad no significa que no le sirva; significa que no se
    la pusieron. Asi que todos los pares cuentan, pero con una confianza que
    crece con el numero de interacciones: ``c = 1 + alfa * r``.

    Escrito a mano y con NumPy, como el resto del curso. Con 110 items y unos
    miles de usuarios tarda unos segundos.
    """
    rng = np.random.default_rng(seed)
    n_u, n_i = len(particion.para_ajustar), particion.n_items

    R = np.zeros((n_u, n_i), dtype=float)
    for u, tr in enumerate(particion.para_ajustar):
        np.add.at(R[u], tr, 1.0)
    P = (R > 0).astype(float)          # preferencia
    C = 1.0 + alfa * R                 # confianza

    X = rng.normal(0, 0.01, (n_u, factores))
    Y = rng.normal(0, 0.01, (n_i, factores))
    I = np.eye(factores)

    for _ in range(pasos):
        YtY = Y.T @ Y
        for u in range(n_u):
            Cu = C[u]
            A = YtY + (Y.T * (Cu - 1.0)) @ Y + regular * I
            b = (Y.T * Cu) @ P[u]
            X[u] = np.linalg.solve(A, b)
        XtX = X.T @ X
        for i in range(n_i):
            Ci = C[:, i]
            A = XtX + (X.T * (Ci - 1.0)) @ X + regular * I
            b = (X.T * Ci) @ P[:, i]
            Y[i] = np.linalg.solve(A, b)

    def hacer(u: int):
        def recomendar(historial):
            puntos = Y @ X[u]
            if not repetir:
                puntos = puntos.copy()
                puntos[np.unique(historial)] = -np.inf
            return [int(i) for i in np.argsort(puntos)[::-1][:k]]
        return recomendar

    # El evaluador pasa historiales, no indices de usuario, asi que se
    # identifica al usuario por su historial. Es feo y es explicito: la
    # alternativa seria cambiar la firma del evaluador solo por este modelo.
    indice = {tuple(int(x) for x in tr): u for u, tr in enumerate(particion.para_ajustar)}

    def recomendar(historial):
        u = indice.get(tuple(int(x) for x in historial))
        if u is None:
            return []
        return hacer(u)(historial)

    return _con_nombre(recomendar, f"factorización ({factores} factores)")


MODELOS = {
    "popularidad": por_popularidad,
    "al azar": al_azar,
    "repetir lo propio": repetir_lo_propio,
    "novedad pura": novedad_pura,
    "kNN por ítems": knn_items,
    "factorización": factorizacion_implicita,
}


# ══════════════════════════════════════════════════════════════════════════
# 6 · arranque en frio
# ══════════════════════════════════════════════════════════════════════════

def usuarios_nuevos(particion: Particion, fraccion: float = 0.2,
                    seed: int = 0) -> Particion:
    """Aparta una fraccion de usuarios para que el modelo **no los vea nunca**.

    Es el arranque en frio, medido en vez de contado. El modelo se ajusta con
    los usuarios que quedan (``ajuste``) y se evalua solo con los apartados
    (``train`` y ``test``), que llegan con su historial pero sin haber estado
    en el ajuste.

    Un modelo que aprende **un vector por usuario** no tiene nada que ofrecer
    aqui, y eso no es un fallo de implementacion: es la forma del modelo. Un
    modelo que calcula el vector del usuario **a partir de su historial** si
    puede. Esa es toda la diferencia entre una factorizacion y una torre.
    """
    rng = np.random.default_rng(seed)
    n = len(particion.train)
    idx = rng.permutation(n)
    corte = max(1, int(n * fraccion))
    nuevos, conocidos = np.sort(idx[:corte]), np.sort(idx[corte:])
    return Particion(
        nombre=f"{particion.nombre} · usuarios nuevos",
        train=[particion.train[u] for u in nuevos],
        test=[particion.test[u] for u in nuevos],
        n_items=particion.n_items,
        ajuste=[particion.train[u] for u in conocidos],
    )


# ══════════════════════════════════════════════════════════════════════════
# 7 · dos modelos mas, los del bloque 4
# ══════════════════════════════════════════════════════════════════════════

def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _pares(historiales: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Todas las parejas (usuario, item observado), aplanadas."""
    positivos = [np.unique(h) for h in historiales]
    us = np.concatenate([np.full(len(p), u, dtype=int) for u, p in enumerate(positivos)])
    its = np.concatenate(positivos).astype(int)
    return us, its


def bpr(particion: Particion, factores: int = 32, pasos: int = 80, lr: float = 0.05,
        regular: float = 0.001, lote: int = 4096, k: int = 50, seed: int = 0,
        repetir: bool = True):
    """Bayesian Personalized Ranking (Rendle y otros, 2009).

    La diferencia con la factorizacion implicita no esta en el modelo, que es
    el mismo producto de dos vectores: esta en **lo que se optimiza**. ALS
    intenta reconstruir la matriz. BPR no intenta reconstruir nada: intenta que
    para cada usuario, **lo que si practico puntue mas alto que lo que no**.

    Eso importa porque la metrica que nos interesa es un orden, no una
    reconstruccion. Un modelo que aproxima bien la matriz puede ordenar mal, y
    a eso se le llama optimizar la perdida equivocada.

    Se entrena con descenso estocastico sobre tripletas (usuario, item
    observado, item no observado).
    """
    rng = np.random.default_rng(seed)
    ajuste = particion.para_ajustar
    n_u, n_i = len(ajuste), particion.n_items

    tiene = np.zeros((n_u, n_i), dtype=bool)
    for u, h in enumerate(ajuste):
        tiene[u, np.unique(h)] = True

    X = rng.normal(0.0, 0.1, (n_u, factores))
    Y = rng.normal(0.0, 0.1, (n_i, factores))
    pares_u, pares_i = _pares(ajuste)

    for _ in range(pasos):
        orden = rng.permutation(len(pares_u))
        for s in range(0, len(orden), lote):
            sel = orden[s:s + lote]
            u, i = pares_u[sel], pares_i[sel]
            j = rng.integers(0, n_i, len(sel))
            valido = ~tiene[u, j]              # el negativo tiene que ser negativo
            u, i, j = u[valido], i[valido], j[valido]
            if len(u) == 0:
                continue
            xu, yi, yj = X[u], Y[i], Y[j]
            dif = np.einsum("bf,bf->b", xu, yi - yj)
            g = _sigmoide(-dif)[:, None]      # derivada de log sigmoide(dif)
            np.add.at(X, u, lr * (g * (yi - yj) - regular * xu))
            np.add.at(Y, i, lr * (g * xu - regular * yi))
            np.add.at(Y, j, lr * (-g * xu - regular * yj))

    indice = {tuple(int(x) for x in h): u for u, h in enumerate(ajuste)}

    def recomendar(historial):
        u = indice.get(tuple(int(x) for x in historial))
        if u is None:
            return []                          # usuario que no estaba: no hay vector
        puntos = Y @ X[u]
        if not repetir:
            puntos = puntos.copy()
            puntos[np.unique(historial)] = -np.inf
        return [int(i) for i in np.argsort(puntos)[::-1][:k]]

    return _con_nombre(recomendar, f"BPR ({factores} factores)")


def dos_torres(particion: Particion, factores: int = 32, oculta: int = 128,
               pasos: int = 25, lr: float = 0.01, lote: int = 256, k: int = 50,
               seed: int = 0, repetir: bool = True):
    """Arquitectura de dos torres, con la misma clase ``Red`` del DQN.

    Dos codificadores que se encuentran en un producto escalar:

    - **La torre del usuario** es un perceptron que recibe el historial (una
      cuenta por item, normalizada) y devuelve un vector. No hay una fila por
      usuario: el vector se **calcula**.
    - **La torre del item** es una tabla de vectores, uno por item.

    Ese detalle, que parece de implementacion, es la razon de que esta
    arquitectura este en produccion en todas partes: como el vector del usuario
    se calcula a partir de su historial, **funciona con un usuario que el
    modelo no ha visto nunca**. Una factorizacion no puede.

    Se entrena con entropia cruzada sobre el catalogo completo. Con 111 items
    eso es exacto y barato; con millones habria que muestrear los negativos, y
    esa es la unica diferencia con la version industrial.
    """
    from .redes import Adam, Red

    rng = np.random.default_rng(seed)
    ajuste = particion.para_ajustar
    n_u, n_i = len(ajuste), particion.n_items

    torre = Red((n_i, oculta, factores), seed=seed)
    E = rng.normal(0.0, 0.1, (n_i, factores))
    opt = Adam(torre.parametros() + [E], lr=lr)

    H = np.zeros((n_u, n_i))
    for u, h in enumerate(ajuste):
        np.add.at(H[u], h, 1.0)
    H /= np.maximum(H.sum(axis=1, keepdims=True), 1.0)

    pares_u, pares_i = _pares(ajuste)

    for _ in range(pasos):
        orden = rng.permutation(len(pares_u))
        for s in range(0, len(orden), lote):
            sel = orden[s:s + lote]
            u, i = pares_u[sel], pares_i[sel]
            U = torre.adelante(H[u])                     # (B, factores)
            logits = U @ E.T                             # (B, n_items)
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            d = p
            d[np.arange(len(u)), i] -= 1.0
            d /= len(u)
            torre.atras(d @ E)
            opt.paso(torre.parametros() + [E], torre.gradientes() + [d.T @ U])

    def recomendar(historial):
        h = np.zeros(n_i)
        np.add.at(h, historial, 1.0)
        total = h.sum()
        if total > 0:
            h /= total
        u = torre.adelante(h)[0]
        puntos = E @ u
        if not repetir:
            puntos = puntos.copy()
            puntos[np.unique(historial)] = -np.inf
        return [int(i) for i in np.argsort(puntos)[::-1][:k]]

    return _con_nombre(recomendar, f"dos torres ({factores} factores)")
