# Curso · Aprendizaje por Refuerzo y Sistemas de Recomendación

Plantilla base del módulo. Especialización en Inteligencia Artificial, UPTC.

---

## Arranque rápido

```bash
git clone <URL-de-tu-repositorio>
cd curso-rl-recsys
uv sync
uv run python scripts/verificar.py
```

Si ves **8 de 8 comprobaciones superadas**, tu entorno está listo.
Si no, el propio guion te dice qué falta y cómo resolverlo.

El instructivo completo, paso a paso y para Windows, macOS y Linux, está en
[`docs/instructivo-entorno.html`](docs/instructivo-entorno.html).

---

## La regla del curso

> **El algoritmo vive en el paquete. El notebook es la bitácora.**

Un notebook puede escribir `from rlrs.dp import value_iteration`.
Un notebook **nunca** define `value_iteration`.

En la mayoría de los programas, un error se manifiesta: el programa se detiene o
aparece un mensaje. En aprendizaje por refuerzo, un error de programación no siempre
impide que el entrenamiento se complete. El programa puede ejecutarse sin advertencias
y producir resultados que, a simple vista, parecen razonables.

Por eso trabajamos con código organizado en módulos, pruebas automáticas,
configuración registrada en archivos y experimentos reproducibles. Los notebooks no
son una herramienta inferior: cumplen un propósito distinto, el de explorar, analizar
y visualizar.

Regla práctica: **si algo se va a ejecutar más de una vez, no pertenece a un
notebook.**

---

## Estructura

| Carpeta | Qué contiene | Quién la toca |
|---|---|---|
| `src/rlrs/` | El paquete: entornos, políticas, algoritmos, métricas | Todo el código reutilizable |
| `tests/` | Pruebas con `pytest` | Se ejecutan en cada entrega |
| `configs/` | Un YAML por experimento | Ningún número mágico en el código |
| `experiments/` | Guiones ejecutables; escriben en `runs/` | Los barridos con semillas |
| `notebooks/` | Análisis y figuras; **solo importan de `src/`** | Exploración e informe |
| `scripts/` | Utilidades, incluida la prueba de paso | Nadie las edita |
| `docs/` | Instructivos y documentación | Solo lectura |

---

## Qué trae ya construido

Una versión mínima pero real de las cuatro capas que usaremos todo el curso:

- **`rlrs.envs.GridWorld`**: un MDP pequeño con interfaz de Gymnasium
  (`reset` / `step`) y su modelo explícito (`transitions`), para poder
  comparar a mano lo que calcula el código.
- **`rlrs.policies`**: el contrato `Policy` (`act(obs) -> acción`) que cumplen
  desde una política aleatoria hasta una red entrenada con PPO.
- **`rlrs.dp.value_iteration`**: iteración de valor, con su prueba.
- **`rlrs.evaluation`**: el arnés, semillas explícitas, intervalos de
  confianza, comparación entre políticas sobre los mismos episodios.

Pruébalo:

```bash
uv run python experiments/run.py --config configs/base.yaml
```

Deberías ver la rejilla resuelta y algo parecido a esto:

```
  optima         retorno +0.636 [+0.603, +0.670]  exito 98.0%  pasos   9.1
  aleatoria      retorno -2.520 [-2.679, -2.361]  exito 25.7%  pasos  56.8
```

Ese contraste es la comprobación de cordura del arnés: si la política óptima
no le gana a la aleatoria, algo está roto en la evaluación, no en el agente.

---

## Comandos que usarás

```bash
uv sync                                    # crear/actualizar el entorno
uv run pytest -q                           # ejecutar las pruebas
uv run ruff check .                        # revisar el estilo
uv run ruff format .                       # formatear
uv run python experiments/run.py           # correr el experimento base
uv run python scripts/verificar.py         # prueba de paso del entorno
uv run jupyter lab                         # abrir los notebooks
```

PyTorch **no** está en las dependencias base, para que la primera instalación
sea rápida y fiable. Se añade en el Bloque 1:

```bash
uv sync --extra dl
```

---

## Antes de cada entrega

1. `uv run pytest -q` en verde.
2. `uv run ruff check .` sin avisos.
3. Los notebooks no definen algoritmos: solo importan de `src/`.
4. Cada resultado del informe se puede regenerar con un comando y su config.
