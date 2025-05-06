# input usuario
from auxiliary import create_forest, get_data
from post_optimization import base_case
from simulator import generate, generate_forest
from use_of_QGIS import create_protection_value_shp, fuels_creation_cortafuegos

# 1 se generan los rodales con maejos
gdf = get_data(".\\test\\data_modificada\\proto_mod.shp")  # se adquiere el shapefile de los rodales

create_forest(gdf, "rid")  # se crea el bosque
RR = generate_forest()  # se generan los rodales

rodales = generate(rodales=RR)  # se generan los rodales con manejos

# 2 obtener datos del bosque sin manejos

rodales_base = base_case(rodales)
fuels_creation_cortafuegos(gdf, rodales_base)  # generar raster de biomasa y fuels por periodo


create_protection_value_shp()  # qeuma y crea un raster con los DPV en valor presente (protection_value.tif)


# 3 en QGIS de utilizar el knapsack polygon para obtener el area de los cortafuegos en distintas capacidades (1%, 2%, 3%)


# 4 calculo sensibilidad
# simular incendios con distintas capacidades de cortafuegos y el caso base sin cortafuegos
# calculadora raster mirar tesis en punto 4.5.2 formula 4.1 para crear archivos expected_loss

# post optimizacion de cortafuegos para calcular la sensibilidad
import matplotlib.pyplot as plt
import numpy as np
from fire2a.raster import read_raster

# Leer los datos de los archivos raster
expected_loss_uno, _ = read_raster(
    "C:\\Users\\xunxo\\OneDrive\\Escritorio\\resultados_comp_sensibilidad\\1%\\expected_loss_1%.tif", info=False
)
expected_loss_dos, _ = read_raster(
    "C:\\Users\\xunxo\\OneDrive\\Escritorio\\resultados_comp_sensibilidad\\2%\\expected_loss_2%.tif", info=False
)
expected_loss_tres, _ = read_raster(
    "C:\\Users\\xunxo\\OneDrive\\Escritorio\\resultados_comp_sensibilidad\\3%\\expected_loss_3%.tif", info=False
)
expected_loss_caso_case, _ = read_raster(
    "C:\\Users\\xunxo\\OneDrive\\Escritorio\\resultados_comp_sensibilidad\\caso_base\\expected_loss_caso_base.tif",
    info=False,
)
# agregar revision que no sea
assert np.all(expected_loss_uno >= 0)
# Reemplazar valores negativos por 0
expected_loss_uno = np.where(expected_loss_uno < 0, 0, expected_loss_uno)
expected_loss_dos = np.where(expected_loss_dos < 0, 0, expected_loss_dos)
expected_loss_tres = np.where(expected_loss_tres < 0, 0, expected_loss_tres)
expected_loss_caso_case = np.where(expected_loss_caso_case < 0, 0, expected_loss_caso_case)

# Calcular las pérdidas y sensibilidades
perdida_uno = np.sum(expected_loss_uno)
perdida_dos = np.sum(expected_loss_dos)
perdida_tres = np.sum(expected_loss_tres)
perdida_caso_base = np.sum(expected_loss_caso_case)

sensibilidad_uno = perdida_caso_base - perdida_uno
sensibilidad_dos = perdida_caso_base - perdida_dos
sensibilidad_tres = perdida_caso_base - perdida_tres

# Preparar los datos para el gráfico
sensibilidades = ["1%", "2%", "3%"]
npe_values = [sensibilidad_uno, sensibilidad_dos, sensibilidad_tres]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(sensibilidades, npe_values, color=["blue", "green", "red"])

# Añadir título y etiquetas
plt.title("Net Protective Effect (NPE)")
plt.xlabel("Capacidades")
plt.ylabel("NPE")

# Mostrar y guardar el gráfico
plt.grid(True)
plt.savefig("net_protective_effect.png")
plt.show()
# escoger capacidad con mejor (mayor) sensibilidad (mirar grafico)
# 5 utilizar estadistica de zona de qgis para tener el mean de cortafuegos por rodal

# output final proto_Cf.shp y cortafuegos.tif
