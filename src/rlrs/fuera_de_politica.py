"""Decidir sin desplegar: IPS, SNIPS y doblemente robusto.

Es la pregunta que se anuncio en la sesion 2 y que lleva seis sesiones
pendiente. Tenemos un registro de lo que hizo el sistema **anterior** y lo que
paso. Tenemos un sistema nuevo. Queremos saber si el nuevo es mejor, y no
podemos desplegarlo para averiguarlo: si lo desplegamos y es peor, el dano ya
esta hecho.

El problema, en una frase
-------------------------
El registro solo dice que paso con **la accion que se tomo**. De la accion que
el sistema nuevo habria tomado no hay nada, porque nunca se mostro. No es que
falte un dato: es que ese dato no existe.

Los cuatro estimadores
----------------------
- **Directo.** Se ajusta un modelo de recompensa con el registro y se pregunta
  cuanto valdria la politica nueva segun ese modelo. Barato, de varianza baja,
  y **sesgado exactamente en lo que el modelo se equivoque**.
- **IPS.** Cada observacion se pesa por ``pi1(a|u) / pi0(a|u)``: si la politica
  nueva habria elegido esa accion mas a menudo que la vieja, esa observacion
  cuenta mas. Es insesgado, y su varianza se dispara cuando las dos politicas
  se parecen poco.
- **SNIPS.** El mismo IPS dividido entre la media de los pesos. Introduce un
  sesgo pequeno y a cambio no explota. Casi siempre es lo que uno quiere.
- **Doblemente robusto.** Suma los dos: parte del modelo de recompensa y corrige
  con IPS solo el error del modelo. Se llama asi porque **acierta si acierta
  cualquiera de los dos**, el modelo o los pesos.

La condicion que no se puede saltar
-----------------------------------
Si la politica de registro **nunca** pudo elegir una accion que la politica
nueva si elige, no hay ningun estimador que lo arregle. No es un problema de
varianza: es que no hay informacion. Por eso las politicas de registro serias
siempre reservan una fraccion de aleatoriedad, y por eso ``politica_registro``
lleva un ``epsilon`` que no se puede poner a cero sin avisar.

Aqui el simulador es nuestro, asi que ademas se puede calcular el **valor
verdadero** de la politica nueva y ver cuanto se equivoco cada estimador. Fuera
del aula eso no se puede, y ese es justo el motivo de que todo esto exista.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# 1 · el mundo y las politicas
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Registro:
    """Lo que dejo escrito la politica anterior.

    Una fila por interaccion. ``propension`` es la probabilidad con la que la
    politica de registro eligio esa accion, y es lo unico que hace posible todo
    lo demas: **un registro sin propensiones no se puede reutilizar**.
    """

    usuarios: np.ndarray
    acciones: np.ndarray
    recompensas: np.ndarray
    propensiones: np.ndarray

    def __len__(self) -> int:
        return len(self.usuarios)


def preferencias(secuencias: list[np.ndarray], n_items: int, suelo: float = 0.02,
                 tope: float = 0.6) -> np.ndarray:
    """Probabilidad de que cada usuario acepte cada item. Es la verdad oculta.

    Sale de la frecuencia con la que cada usuario practico cada habilidad en su
    historial completo. El ``tope`` evita que exista una accion que se acepta
    siempre, que haria el problema trivial.
    """
    P = np.zeros((len(secuencias), n_items))
    for u, s in enumerate(secuencias):
        np.add.at(P[u], s, 1.0)
    P = P / np.maximum(P.max(axis=1, keepdims=True), 1.0)
    return suelo + (tope - suelo) * P


def _fila_softmax(puntos: np.ndarray, temperatura: float) -> np.ndarray:
    z = puntos / max(temperatura, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def politica_registro(pref: np.ndarray, temperatura: float = 1.0,
                      epsilon: float = 0.1, catalogo: int | None = None) -> np.ndarray:
    """La politica que genero los datos: popular, tibia, y con exploracion.

    No mira al usuario: reparte segun lo popular en la poblacion. Es lo que
    suele haber en un sistema anterior. El ``epsilon`` mezcla una uniforme, y
    eso garantiza que **toda accion tiene probabilidad positiva**, que es la
    condicion de soporte.
    """
    if epsilon <= 0.0 and catalogo is None:
        raise ValueError(
            "epsilon = 0 deja acciones con probabilidad cero, y un registro asi "
            "no se puede reutilizar: ningun estimador lo arregla. Si lo que "
            "quiere es DEMOSTRARLO, pase `catalogo=` para recortar el catalogo "
            "a proposito.")
    popular = pref.mean(axis=0, keepdims=True)
    base = _fila_softmax(np.repeat(popular, len(pref), axis=0), temperatura)
    uniforme = np.full_like(base, 1.0 / pref.shape[1])
    pi = (1.0 - epsilon) * base + epsilon * uniforme
    if catalogo is not None:
        # El sistema anterior solo tenia estos items en el catalogo. Todo lo
        # demas es literalmente inobservable: no hay dato, y no lo va a haber.
        vivos = np.argsort(popular[0])[::-1][:catalogo]
        mascara = np.zeros(pref.shape[1], dtype=bool)
        mascara[vivos] = True
        pi = pi * mascara
        pi = pi / pi.sum(axis=1, keepdims=True)
    return pi


def politica_personal(pref: np.ndarray, temperatura: float = 0.3,
                      epsilon: float = 0.0) -> np.ndarray:
    """La politica nueva: mira al usuario. Cuanto menor la temperatura, mas decidida."""
    base = _fila_softmax(pref, temperatura)
    if epsilon > 0.0:
        base = (1.0 - epsilon) * base + epsilon / pref.shape[1]
    return base


def registrar(pref: np.ndarray, pi0: np.ndarray, n: int, seed: int = 0) -> Registro:
    """Corre la politica de registro y anota que paso. Recompensa 0 o 1."""
    rng = np.random.default_rng(seed)
    n_u, n_i = pref.shape
    us = rng.integers(0, n_u, n)
    acumulada = np.cumsum(pi0[us], axis=1)
    sorteo = rng.random((n, 1))
    acciones = (sorteo > acumulada).sum(axis=1).clip(0, n_i - 1)
    p = pref[us, acciones]
    recompensas = (rng.random(n) < p).astype(float)
    return Registro(us, acciones, recompensas, pi0[us, acciones])


def valor_verdadero(pref: np.ndarray, pi: np.ndarray) -> float:
    """Lo que de verdad rinde una politica. Solo se puede calcular en un simulador."""
    return float((pi * pref).sum(axis=1).mean())


# ══════════════════════════════════════════════════════════════════════════
# 2 · los estimadores
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Estimacion:
    nombre: str
    valor: float
    ic: float
    verdad: float = float("nan")

    @property
    def error(self) -> float:
        return abs(self.valor - self.verdad)

    @property
    def cubre(self) -> bool:
        """Si el intervalo del estimador contiene el valor verdadero."""
        return abs(self.valor - self.verdad) <= self.ic

    def __str__(self) -> str:
        marca = "sí" if self.cubre else "NO"
        return (f"{self.nombre:<22s} {self.valor:>8.4f} ± {self.ic:<7.4f} "
                f"error {self.error:>7.4f}   cubre la verdad: {marca}")


def _ic95(muestras: np.ndarray) -> float:
    return float(1.96 * np.std(muestras, ddof=1) / np.sqrt(len(muestras)))


def modelo_de_recompensa(registro: Registro, n_items: int, suavizado: float = 1.0) -> np.ndarray:
    """Recompensa media observada por accion, suavizada. **Ignora al usuario a proposito.**

    Es un modelo pobre, y esta elegido asi. El metodo directo va a heredar ese
    error entero, y el doblemente robusto lo va a corregir. Con un modelo
    perfecto no se veria la diferencia entre los dos y la clase no ensenaria
    nada.
    """
    suma = np.full(n_items, suavizado * 0.5)
    cuenta = np.full(n_items, suavizado)
    np.add.at(suma, registro.acciones, registro.recompensas)
    np.add.at(cuenta, registro.acciones, 1.0)
    return suma / cuenta


def directo(registro: Registro, pi1: np.ndarray, r_gorro: np.ndarray) -> Estimacion:
    """Metodo directo: se cree el modelo de recompensa y no mira el registro."""
    por_usuario = pi1 @ r_gorro
    return Estimacion("directo", float(por_usuario.mean()), _ic95(por_usuario))


def _pesos(registro: Registro, pi1: np.ndarray) -> np.ndarray:
    return pi1[registro.usuarios, registro.acciones] / registro.propensiones


def ips(registro: Registro, pi1: np.ndarray) -> Estimacion:
    """Muestreo por importancia. Insesgado, y con la varianza que le toque."""
    v = _pesos(registro, pi1) * registro.recompensas
    return Estimacion("IPS", float(v.mean()), _ic95(v))


def snips(registro: Registro, pi1: np.ndarray) -> Estimacion:
    """IPS normalizado por la media de los pesos. Un poco sesgado, mucho mas estable."""
    w = _pesos(registro, pi1)
    v = w * registro.recompensas
    media_w = max(w.mean(), 1e-12)
    return Estimacion("SNIPS", float(v.mean() / media_w), _ic95(v / media_w))


def doblemente_robusto(registro: Registro, pi1: np.ndarray,
                       r_gorro: np.ndarray) -> Estimacion:
    """Modelo de recompensa mas correccion por importancia solo del residuo."""
    w = _pesos(registro, pi1)
    base = (pi1 @ r_gorro)[registro.usuarios]
    residuo = registro.recompensas - r_gorro[registro.acciones]
    v = base + w * residuo
    return Estimacion("doblemente robusto", float(v.mean()), _ic95(v))


def tamano_efectivo(registro: Registro, pi1: np.ndarray) -> float:
    """Cuantas observaciones valen de verdad los pesos que tenemos.

    ``(suma de w)^2 / suma de w^2``. Si de diez mil observaciones el tamano
    efectivo es doscientas, la cifra que salga tendra la precision de doscientas
    observaciones por mucho que el informe diga diez mil. Es el diagnostico que
    hay que mirar **antes** que la estimacion.
    """
    w = _pesos(registro, pi1)
    s = w.sum()
    return float(s * s / max((w ** 2).sum(), 1e-12))


# ══════════════════════════════════════════════════════════════════════════
# 3 · el experimento completo
# ══════════════════════════════════════════════════════════════════════════

def comparar(pref: np.ndarray, pi0: np.ndarray, pi1: np.ndarray,
             n: int = 20000, seed: int = 0) -> tuple[list[Estimacion], float, float]:
    """Registra con ``pi0``, estima el valor de ``pi1`` de cuatro formas y compara.

    Devuelve las cuatro estimaciones (con el valor verdadero ya dentro), el
    valor verdadero, y el tamano efectivo de la muestra.
    """
    registro = registrar(pref, pi0, n, seed=seed)
    r_gorro = modelo_de_recompensa(registro, pref.shape[1])
    verdad = valor_verdadero(pref, pi1)

    ests = [directo(registro, pi1, r_gorro),
            ips(registro, pi1),
            snips(registro, pi1),
            doblemente_robusto(registro, pi1, r_gorro)]
    for e in ests:
        e.verdad = verdad
    return ests, verdad, tamano_efectivo(registro, pi1)


def masa_observable(pi0: np.ndarray, pi1: np.ndarray) -> float:
    """Que fraccion de lo que haria la politica nueva pudo llegar a observarse.

    Es la comprobacion de soporte, y es de una linea. Si sale menos de 1, hay
    acciones que la politica nueva toma y que el registro **nunca** pudo
    mostrar: sobre esas no hay dato y no lo va a haber.

    Conviene mirar esto ANTES que el tamano efectivo, porque el tamano efectivo
    no detecta este fallo: los pesos que existen pueden estar perfectamente
    repartidos mientras falta la mitad del problema.
    """
    return float((pi1 * (pi0 > 0)).sum(axis=1).mean())


def distancia(pi0: np.ndarray, pi1: np.ndarray) -> float:
    """Distancia media entre las dos politicas, en variacion total.

    0 es la misma politica, 1 es que no coinciden en nada. Es el numero que
    predice si los estimadores van a funcionar, y conviene mirarlo antes.
    """
    return float(0.5 * np.abs(pi1 - pi0).sum(axis=1).mean())
