#!python
"""Este codigo es el encargado de correr todos los pasos de los procesos del README 

"""
def simular_crecimiento():
    """paso 1, simular crecimiento de rodales y crear el bosque

    si se requiere bosque al azar agregar "--random" en simulator.main

    simulator y auxiliary son modulos de https://github.com/fire2a/growth

    inputs: 
       - area de estudio (shapefile)
       - archivo de configuracion (config.toml)
    outputs:
       - bosque_data.csv (csv con los rodales y sus atributos)
       - gdf (geopandas dataframe con los rodales y sus atributos)
       - rodales (lista de diccionarios con los rodales y sus atributos)
    """
    
    from auxiliary import get_data, create_forest
    import simulator
    from pathlib import Path

    area_estudio=Path("test/data_modificada/proto_mod.shp")
    gdf = get_data(area_estudio)

    create_forest(gdf, id="rid",outfile="bosque_data.csv")
    rodales = simulator.main(["config.toml", "-m", "tabla.csv","-d", "bosque_data.csv","-s"]) 
    
    return gdf, rodales

gdf, rodales = simular_crecimiento() # paso 1

def crear_opciones_cortafuegos(gdf,rodales): #paso 2
    # solo para linux ssh necesario paso 1

    # import os
    # os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from simulator import read_toml
    from post_optimization import base_case
    from use_of_QGIS import fuels_creation_cortafuegos, create_protection_value_shp
    
    config = read_toml("config.toml")  # leer el archivo de configuración
    config_opti = read_toml("config_opti.toml")  # leer el archivo de configuración
    rodales_base = base_case(rodales)
    print("fuels se guardan en carpeta de cortafuegos")
    fuels_creation_cortafuegos(gdf, rodales_base,config)  # generar raster de biomasa y fuels por periodo 
    create_protection_value_shp(config,config_opti)  # quema y crea un raster con los DPV en valor presente (protection_value.tif)
    print("Se han creado los DPV para decidir cortafuegos, se guardan en la carpeta de cortafuegos")
crear_opciones_cortafuegos(gdf,rodales)

#paso 3 es calcular luego para elegir cortafuegos optimizar knapsack con valor dpv restringiendo 1,2 y 3% de area tratada (cortafuego)
#terminar calculando expected losses de cada cortafuego y caso base, para eso es debido simular incendios con cada cortafuego y caso base, luego ocupar calculadora raster
def calcular_sensibilidad_cortafuegos(caso_base_path,cortafuego_1_path,cortafuego_2_path,cortafuego_3_path) #paso 4
    from fire2a.raster import read_raster
    import matplotlib.pyplot as plt
    import numpy as np
    # simular incendios con distintas capacidades de cortafuegos y el caso base sin cortafuegos
    # calculadora raster mirar tesis en punto 4.5.2 formula 4.1 para crear archivos expected_loss


    # Leer los datos de los archivos raster
    expected_loss_uno, _ = read_raster(cortafuego_1_path, info=False)
    expected_loss_dos, _ = read_raster(cortafuego_2_path, info=False)
    expected_loss_tres, _ = read_raster(cortafuego_3_path, info=False)
    expected_loss_caso_case, _ = read_raster(caso_base_path, info=False)
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

