# input usuario
from tactico import *
from simulator import *
from post_optimization import *
from use_of_QGIS import *
import os

# gen bosque
rodales = generate()
write(rodales)
politicas = print_manejos_possibles()

# optimiza el modelo 1 sin incendios
q = model_t(rodales, politicas, config["opti"]["Price"])
# filtra los datos de los manejos escogidos
f = filtro(rodales)
# crea matrices de los fuels
m = matriz_cod(f, 6, 6)
# crea los fuels.asc dependiendo de la solucion y periodo
# Guardar cada matriz en un archivo .asc
directorio_salida = "./soluciones/"  # Cambia esto si necesitas otro directorio

# Crear el directorio si no existe
if not os.path.exists(directorio_salida):
    os.makedirs(directorio_salida)
for s, solucion in enumerate(m):
    for t, matriz in enumerate(solucion):
        nombre_archivo = os.path.join(directorio_salida, f"fuels_solucion_{s}_periodo_{t}.asc")
        matriz_a_asc(matriz, nombre_archivo)


bp = burn_prob_sol(5)
new_biomass = multiplicar_listas(bp, f)
biomass_for_solution = sumar_por_solucion(new_biomass)
print(biomass_for_solution)
# simulas los incendios

# get bp
