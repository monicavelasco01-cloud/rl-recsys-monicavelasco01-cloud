from rlrs.dqn import Memoria
from rlrs.redes import Red
from rlrs import GridWorld
from rlrs.redes import Adam 
import numpy as np

# 1. CONTROL 1: MEMORIA

# 1.1 Definimos los parámetros del entorno
capacidad = 10

# 1.2 Instanciamos nuestra Memoria
mem = Memoria(capacidad, 4)

# 1.3 Guardamos 12 transiciones que son 2 más de la capacidad

for i in range(12):
    mem.guardar(np.zeros(4), 0, 0.0, np.zeros(4), False)

# 1.4 Comprobamos el Control 1
print("\n--- Control 1: Memoria ---")
print(f"Capacidad máxima: {capacidad}")
print(f"Elementos guardados actualmente: {len(mem)}")
print(f"Índice interno actual (_i): {mem._i}")


#CONTROL 2: LOTE

#2.1 Extraemos un lote de tamaño 5
n_lote = 5
rgn = np.random.default_rng(seed=42)
# Usamos el método .lote() de la clase Memoria
s_lote, a_lote, r_lote, s2_lote, fin_lote = mem.lote(n_lote, rgn)

print("\n --- Control 2: Extracción de Lote ---")
print(f"Forma de s_lote: {s_lote.shape}")
print(f"Forma de a_lote: {a_lote.shape}")
print(f"¿Son los 5 arreglos del mismo tamaño?: {len(s_lote) == len(a_lote) == len(r_lote) == len(s2_lote) == len(fin_lote)}")

# Control 3: Red

#3.1 Definimos la arquitectura: 4 entradas, 64 ocultas, 4 salidas
d = 4
oculta = 64
n_acciones = 4
red = Red((d, oculta, n_acciones), seed=0)

#3.2 Probamos una sola entrada, es decir, un estado traducido.
x= np.array([1.0, 0.0, 0.0, 0.5556]) #estado 0
q_valores = red.adelante(x)

print("\n--- Control 3: Red ---")
print(f"Valores Q para el estado 0: {q_valores}")
print(f"Forma de la salida: {q_valores.shape}")

#3.3 Probamos con el lote que extrajimos antes (s_lote)
q_lote = red.adelante(s_lote)
print(f"Forma de la salida para el lote: {q_lote.shape}")

# Control 4: Aprendizaje (Actualización de los pesos), por medio de la ecuación de Bellman

#4.1 Parametros de Aprendizaje
gamma = 0.99

#4.2 Calculamos valores Q para el estado siguiente (s2)
q_s2 = red.adelante(s2_lote)

#4.3 Obtenemos el máximo valor Q para cada transición en el lote
max_q_s2 = np.max(q_s2, axis=1)

#4.4 Calculamos el Target (meta segun Bellman)
target = r_lote + gamma * max_q_s2 * (1 - fin_lote)

print("\n --- Control 4: Aprendizaje (Target) ---")
print(f"Valores de recompensa (r) : {r_lote}")
print(f"Valores objetivo (target): {target}")


# Control 5: Gradiente

#5.1 obtenemos las predicciones actuales de la red para el lote entero
q_actuales = red.adelante(s_lote)

#5.2 creamos una copia de las predicciones para modificar solo la accion tomada
#ya que solo quiero ajustar el valor Q de la acción que el agente realmente eligio
targets_completos = q_actuales.copy()

#5.3 insertamos el target segun la accion a_lote
for i in range(n_lote):
    targets_completos[i, a_lote[i]] = target [i]

#5.4 Calculamos el Gradiente (Error)
grad = targets_completos - q_actuales  #(prediccion - target)

print("\n--- Control 5: Gradiente ---")
print(f"Gradiente (primeras 2 filas): \n{grad[:2]}")

# Control 6: Bucle
