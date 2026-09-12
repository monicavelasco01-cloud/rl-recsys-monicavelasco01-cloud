"""Reto de la sesion 7 · El usuario que llega manana.

═══════════════════════════════════════════════════════════════════════════
QUE PASA
═══════════════════════════════════════════════════════════════════════════

Esto es lo que sale con particion temporal, y ya lo conocen del reto 6:

    recomendador                 Recall@10   nDCG@10
    dos torres                      0,7456    0,5764
    BPR                             0,7450    0,5734
    factorizacion (ALS)             0,7179    0,5652
    repetir lo propio               0,7009    0,5880

Ahora la misma tabla, pero evaluando con **usuarios que el modelo no vio
nunca**: se ajusta con 2.473 estudiantes y se mide con otros 618 distintos.

    recomendador                 Recall@10   nDCG@10   sin respuesta
    dos torres                      0,7218    0,5768           0,0 %
    repetir lo propio               0,7056    0,5885           0,0 %
    factorizacion (ALS)             0,1537    0,1488          84,1 %
    BPR                             0,1572    0,1504          84,1 %

Lea la ultima columna. **Al 84 % de los usuarios nuevos, la factorizacion no
les devuelve nada.** No es un fallo del codigo: es la forma del modelo. Una
factorizacion aprende un vector por usuario, y de un usuario que no estaba en
la matriz no hay vector que sacar.

El 16 % restante recibe las recomendaciones de otra persona que resulto tener
exactamente el mismo historial. Eso tambien es la forma del modelo.

═══════════════════════════════════════════════════════════════════════════
QUE HAY QUE HACER
═══════════════════════════════════════════════════════════════════════════

Dos cosas, y la primera se entrega aunque la segunda no salga.

1. **El diagnostico, escrito en su bitacora antes de tocar el codigo.** Por que
   la factorizacion no puede responderle a un usuario nuevo. Que tiene la
   arquitectura de dos torres que se lo permite. Y una prediccion con numero:
   cuanto cree que va a sacar su recomendador en cada una de las dos columnas.

2. **El recomendador.** Rellene ``mi_recomendador`` para pasar el liston **en
   las dos condiciones a la vez**:

       Recall@10 >= 0,7100  con particion temporal normal
       Recall@10 >= 0,6500  con usuarios que el modelo no vio nunca

   Mire la tabla otra vez antes de escribir nada. La factorizacion pasa el
   primero y se estrella en el segundo. Repetir lo propio aguanta el segundo y
   se queda corto en el primero.

   Combinar los dos es una respuesta legitima, y **midala antes de creersela**:
   una fusion de las dos listas no siempre gana a la mejor de las dos, y
   descubrir eso con su propio codigo vale mas que aprobar. Mire las dos
   columnas de nDCG del arnes mientras lo intenta.

   Tiene todo ``rlrs.recomendacion`` disponible, incluidas ``dos_torres`` y
   ``bpr``, y puede escribir el suyo desde cero.

═══════════════════════════════════════════════════════════════════════════
COMO SE COMPRUEBA
═══════════════════════════════════════════════════════════════════════════

    uv run python scripts/reto7.py
    uv run pytest tests/test_reto7.py

El arnes mide las dos condiciones y ademas ensena la fraccion de usuarios a
los que su recomendador no supo que decirles. Esa cifra es la que hay que
llevar a la bitacora.

═══════════════════════════════════════════════════════════════════════════
LO QUE NO VALE
═══════════════════════════════════════════════════════════════════════════

Mirar ``particion.test``. El arnes lo comprueba y la prueba falla. No es por
desconfianza: es que la forma mas facil de aprobar cualquiera de estos retos
es mirar la respuesta sin darse cuenta.
"""

from __future__ import annotations

from rlrs.recomendacion import Particion, factorizacion_implicita


def mi_recomendador(particion: Particion, k: int = 50):
    """Devuelva una funcion que reciba un historial y devuelva items ordenados.

    Lo que trae por defecto es la factorizacion implicita, que es el mejor de
    los metodos clasicos sobre la particion normal **y el que peor se comporta
    con un usuario nuevo**. Ejecute el arnes antes de cambiar nada: el primer
    resultado que va a ver es un suspenso, y esta puesto a proposito.
    """
    return factorizacion_implicita(particion, factores=32, pasos=10, k=k)
