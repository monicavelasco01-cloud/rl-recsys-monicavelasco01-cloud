# Carpeta del torneo ciego · sesión 8

Aquí van los recomendadores de los seis equipos, uno por archivo:

    torneo/grupo1.py
    torneo/grupo2.py
    ...
    torneo/grupo6.py

Cada archivo es el `retos/reto7.py` de ese equipo, renombrado. Tiene que
exponer `mi_recomendador(particion, k=50)`, que es la firma del reto.

Se ejecuta así:

    uv run python scripts/torneo.py --sortear --seed 7

El protocolo sale del sorteo: la partición, la k y la métrica que manda.
**No se anuncia antes de la sesión.** Si se sabe de antemano, cada equipo lo
optimiza durante la semana y el torneo deja de medir lo que queríamos medir.

Un archivo que no ejecuta no tumba el torneo: aparece en la tabla con «no
ejecuta», que también es un resultado.
