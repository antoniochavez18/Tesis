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

    """paso 2, crear los DPV en valor presente, fuels y biomasa del bosque sin manejo
    inputs: 
       - area de estudio (gdf)
       - archivo de configuracion (config.toml)
       - archivo de configuracion  de optimizacion (config_opti.toml)
       - rodales (lista de diccionarios con los rodales y sus atributos)
    outputs:
       - DPV en valor presente (raster)
       - biomasa (raster)
       - fuels (raster)
    """
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

def calcular_sensibilidad_cortafuegos(): #paso 3
    """paso 3, crear cortafuegos, calcular la sensibilidad (NPE) de los cortafuegos y guardar el mejor cortafuegos

    inputs: 
       - DPV en valor presente (raster)
       - biomasa (raster)
       - fuels (raster)
       - lista capacidades de los cortafuegos (cuanto del paisaje es cortafuegos)
       - cordenada (EPSG)
    outputs:
       - archivo tif del cortafuego ganador
    """
    from use_of_QGIS import sensibilidades_cortafuegos
    from pathlib import Path

    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    cordenada = "EPSG:32718"
    capacidades = [0.01, 0.02, 0.03]
    path_DPV = str(Path("./cortafuegos/protection_value.tif"))
    path_fuels = str(Path("./cortafuegos/fuels/fuels_base_periodo_0.tif"))
    path_biomass = str(Path("./cortafuegos/biomass/biomass_base_periodo_0.tif"))
    sensibilidades_cortafuegos(path_DPV,capacidades,path_fuels,path_biomass,cordenada)


calcular_sensibilidad_cortafuegos()

def rodales_con_cortafuegos(rodales): # paso 3.5
    """paso 3.5, crear el paisaje con cortafuegos y calcular la biomasa con cortafuegos
    inputs: 
       - rodales (lista de diccionarios con los rodales y sus atributos)
       - cortafuegos (raster)
    outputs:
       - rodales con cortafuegos (lista de diccionarios con los rodales y sus atributos)
    """
    from use_of_QGIS import create_paisaje_con_cortafuegos
    from auxiliary import get_data
    from post_optimization import biomass_with_fire_breacks
    from pathlib import Path
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    area_estudio=str(Path("test/data_modificada/proto_mod.shp"))
    cortafuegos = str(Path("cortafuegos/cortafuegos_0.02.tif"))
    create_paisaje_con_cortafuegos(area_estudio,cortafuegos) # crear paisaje con cortafuegos
    area_con_cortafuegos= Path("cortafuegos/data_cortafuegos/data_modificada/proto_mod.shp")
    gdf_cf = get_data(area_con_cortafuegos)  # se adquiere el shapefile rodales con cortafuegos
    gdf_cf = gdf_cf.sort_values(by="rid")

    # se renombra la columna _mean a prop_cf (proporcion de cortafuegos) mean porque venia de estadistica zonal
    gdf_cf.rename(columns={"_mean": "prop_cf"}, inplace=True)
    rodales_cf = biomass_with_fire_breacks(
        rodales, gdf_cf, "rid"
    )
    return rodales_cf  # leer el paisaje con cortafuegos

rodales_cf = rodales_con_cortafuegos(rodales)

