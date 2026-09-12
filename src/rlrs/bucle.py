"""El bucle de realimentacion: cuando el recomendador se come sus propios datos.

Hasta aqui, todo el bloque 3 midio recomendadores sobre un registro FIJO. Se
ajusta con el pasado, se evalua con el futuro, y el futuro estaba escrito antes
de que llegaramos. Eso no es lo que pasa en produccion.

En produccion el recomendador **decide que se muestra**, y lo que se muestra
decide que se puede observar, y lo observado es lo que entrena a la version
siguiente. El sistema se alimenta de datos que el mismo genero.

Este modulo cierra ese bucle:

    ronda 1:  el recomendador propone  ->  el usuario acepta algo
              lo aceptado entra en el registro
    ronda 2:  el recomendador se REAJUSTA con el registro ampliado
              propone otra vez...

y mide, ronda a ronda, cuanto catalogo queda vivo.

Lo que hay que mirar no es el acierto, que sube. Es la **cobertura**, que baja,
y la **concentracion**, que sube. Nadie cambio una linea de codigo.

Sobre el simulador de usuarios
------------------------------
Cada usuario simulado es un usuario real de ASSISTments, y su preferencia
latente es la frecuencia con la que practico cada habilidad **en todo su
historial**, incluido el tramo que los recomendadores no ven.

La preferencia lleva un ``suelo``: ningun item tiene probabilidad cero de ser
aceptado. Ese detalle no es cosmetico. Sin suelo, el bucle se cierra solo por
construccion y la demostracion no demuestra nada: bastaria decir que lo que
nunca se acepta nunca entra. Con suelo, **todo item podria entrar en el
registro**, y si aun asi el catalogo se encoge, el encogimiento es del bucle y
no del simulador.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .recomendacion import Datos, Particion

CIMA = 10          # cuantos items se miran para la concentracion


@dataclass
class Ronda:
    """Lo que pasa en una vuelta del bucle."""

    ronda: int
    cobertura: float          # fraccion del catalogo recomendada en esta ronda
    items_distintos: int
    concentracion: float      # fraccion de recomendaciones que se llevan los 10 primeros
    catalogo_efectivo: float  # exp(entropia) de lo recomendado: items usados de verdad
    homogeneidad: float       # parecido medio entre los registros de dos usuarios
    novedad: float            # sorpresa media respecto al registro ACTUAL
    aceptacion: float         # fraccion de lo recomendado que el usuario acepto
    registro: int             # tamano del registro al terminar la ronda

    def __str__(self) -> str:
        return (f"  {self.ronda:>5}  {self.cobertura:>9.3f}  {self.catalogo_efectivo:>10.1f}  "
                f"{self.concentracion:>13.3f}  {self.homogeneidad:>12.4f}  "
                f"{self.novedad:>8.2f}  {self.aceptacion:>10.3f}  {self.registro:>9,}")


def _preferencias(secuencias: list[np.ndarray], n_items: int, suelo: float) -> np.ndarray:
    """Preferencia latente por usuario, normalizada a un maximo de 1, con suelo."""
    P = np.zeros((len(secuencias), n_items))
    for u, s in enumerate(secuencias):
        np.add.at(P[u], s, 1.0)
    cima = np.maximum(P.max(axis=1, keepdims=True), 1.0)
    P = P / cima
    return suelo + (1.0 - suelo) * P


def simular(datos: Datos, constructor, rondas: int = 10, usuarios: int = 500,
            k: int = 10, fraccion: float = 0.8, suelo: float = 0.05,
            sesgo_posicion: bool = True, seed: int = 0) -> list[Ronda]:
    """Cierra el bucle durante ``rondas`` vueltas y devuelve una fila por vuelta.

    Parameters
    ----------
    constructor:
        Cualquiera de los de ``rlrs.recomendacion``: recibe una ``Particion`` y
        devuelve una funcion que recibe un historial y devuelve items.
    usuarios:
        Cuantos usuarios simular. Con 500 la vuelta completa tarda segundos y
        el efecto ya se ve. Es un parametro y no un numero escondido porque
        cambiar el tamano de la poblacion cambia la velocidad del encogimiento.
    suelo:
        Probabilidad minima de aceptar cualquier item. Ver la nota del modulo.
    sesgo_posicion:
        Si es cierto, la probabilidad de aceptar se descuenta por la posicion
        con ``1 / log2(posicion + 2)``, que es la forma habitual de decir que
        lo de arriba se mira mas. Es el mecanismo por el que un recomendador
        **fabrica** exposicion, y apagarlo es la mejor ablacion de este
        experimento.
    """
    rng = np.random.default_rng(seed)
    n_items = datos.n_items

    elegidos = rng.permutation(len(datos.secuencias))[:usuarios]
    secuencias = [datos.secuencias[u] for u in elegidos]
    pref = _preferencias(secuencias, n_items, suelo)

    # El registro arranca con el mismo tramo que ve cualquier particion
    # temporal del curso, para que la ronda 0 sea comparable con todo lo demas.
    registro = [s[:max(1, int(len(s) * fraccion))].copy() for s in secuencias]

    descuento = (1.0 / np.log2(np.arange(k) + 2.0)) if sesgo_posicion else np.ones(k)
    filas: list[Ronda] = []

    for r in range(1, rondas + 1):
        particion = Particion(nombre="bucle", train=registro,
                              test=[np.array([], dtype=int) for _ in registro],
                              n_items=n_items)
        recomendar = constructor(particion)

        cuenta_recomendados = np.zeros(n_items)
        propuestas = aceptadas = 0

        # Popularidad del registro ACTUAL: la novedad se mide contra lo que el
        # sistema ya sabe, y lo que sabe cambia en cada vuelta.
        cuenta_registro = np.zeros(n_items)
        for h in registro:
            np.add.at(cuenta_registro, h, 1)
        p_item = np.clip(cuenta_registro / max(cuenta_registro.sum(), 1.0), 1e-12, None)
        sorpresas = []

        nuevos: list[list[int]] = []
        for u, h in enumerate(registro):
            top = [int(i) for i in recomendar(h)][:k]
            if not top:
                nuevos.append([])
                continue
            np.add.at(cuenta_recomendados, top, 1)
            sorpresas.append(float(np.mean(-np.log2(p_item[top]))))
            prob = pref[u, top] * descuento[:len(top)]
            acepta = rng.random(len(top)) < prob
            nuevos.append([i for i, si in zip(top, acepta) if si])
            propuestas += len(top)
            aceptadas += int(acepta.sum())

        for u, add in enumerate(nuevos):
            if add:
                registro[u] = np.concatenate([registro[u], np.array(add, dtype=int)])

        total_rec = max(cuenta_recomendados.sum(), 1.0)
        cima = np.sort(cuenta_recomendados)[::-1][:CIMA].sum()
        q = cuenta_recomendados / total_rec
        vivos = q[q > 0]
        efectivo = float(np.exp(-(vivos * np.log(vivos)).sum())) if len(vivos) else 0.0
        filas.append(Ronda(
            ronda=r,
            cobertura=float((cuenta_recomendados > 0).sum()) / n_items,
            items_distintos=int((cuenta_recomendados > 0).sum()),
            concentracion=float(cima / total_rec),
            catalogo_efectivo=efectivo,
            homogeneidad=_homogeneidad(registro, n_items),
            novedad=float(np.mean(sorpresas)) if sorpresas else 0.0,
            aceptacion=aceptadas / propuestas if propuestas else 0.0,
            registro=int(sum(len(h) for h in registro)),
        ))

    return filas


def cabecera() -> str:
    return (f"  {'ronda':>5}  {'cobertura':>9}  {'catálogo':>10}  "
            f"{'concentración':>13}  {'homogeneidad':>12}  {'novedad':>8}  "
            f"{'aceptación':>10}  {'registro':>9}")


def _homogeneidad(registro: list[np.ndarray], n_items: int) -> float:
    """Parecido medio entre los registros de dos usuarios cualesquiera.

    Es la medida del efecto que de verdad describe la literatura del bucle de
    realimentacion: no es tanto que el catalogo se apague, es que **los
    usuarios se parecen cada vez mas entre si**, porque a todos se les ensena
    lo mismo. Coseno medio sobre todas las parejas.
    """
    M = np.zeros((len(registro), n_items))
    for u, h in enumerate(registro):
        np.add.at(M[u], h, 1.0)
    normas = np.linalg.norm(M, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    M = M / normas
    S = M @ M.T
    n = len(registro)
    return float((S.sum() - np.trace(S)) / (n * (n - 1))) if n > 1 else 0.0


def encogimiento(filas: list[Ronda]) -> float:
    """Cuanto cayo la cobertura entre la primera ronda y la ultima, en tanto por uno."""
    if len(filas) < 2 or filas[0].cobertura == 0:
        return 0.0
    return (filas[0].cobertura - filas[-1].cobertura) / filas[0].cobertura
