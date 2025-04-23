#!python
"""Este codigo es el encargado de correr todos los pasos de los procesos del README"""


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

    from pathlib import Path

    import simulator
    from auxiliary import create_forest, get_data

    area_estudio = Path("test/data_modificada/proto_mod.shp")
    gdf = get_data(area_estudio)

    create_forest(gdf, id="rid", outfile="bosque_data.csv")
    rodales = simulator.main(["config.toml", "-m", "tabla.csv", "-d", "bosque_data.csv", "-s"])

    return gdf, rodales


gdf, rodales = simular_crecimiento()  # paso 1

# crear carpete biomass y fuels guardarlas en carpeta cortafuegos


def crear_opciones_cortafuegos(gdf, rodales):  # paso 2
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
    from post_optimization import base_case
    from simulator import read_toml
    from use_of_QGIS import create_protection_value_shp, fuels_creation_cortafuegos

    config = read_toml("config.toml")  # leer el archivo de configuración
    config_opti = read_toml("config_opti.toml")  # leer el archivo de configuración
    rodales_base = base_case(rodales)
    print("fuels se guardan en carpeta de cortafuegos")
    fuels_creation_cortafuegos(gdf, rodales_base, config)  # generar raster de biomasa y fuels por periodo
    create_protection_value_shp(
        config, config_opti
    )  # quema y crea un raster con los DPV en valor presente (protection_value.tif)
    print("Se han creado los DPV para decidir cortafuegos, se guardan en la carpeta de cortafuegos")


crear_opciones_cortafuegos(gdf, rodales)


def calcular_sensibilidad_cortafuegos():  # paso 3
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
    import os
    from pathlib import Path

    from use_of_QGIS import sensibilidades_cortafuegos

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    cordenada = "EPSG:32718"
    capacidades = [0.01, 0.02, 0.03]
    path_DPV = str(Path("./cortafuegos/protection_value.tif"))
    path_fuels = str(Path("./cortafuegos/fuels/fuels_base_periodo_0.tif"))
    path_biomass = str(Path("./cortafuegos/biomass/biomass_base_periodo_0.tif"))
    sensibilidades_cortafuegos(path_DPV, capacidades, path_fuels, path_biomass, cordenada)


calcular_sensibilidad_cortafuegos()


def rodales_con_cortafuegos(rodales):  # paso 4
    """paso 4, crear el paisaje con cortafuegos y calcular la biomasa con cortafuegos
    inputs:
       - rodales (lista de diccionarios con los rodales y sus atributos)
       - cortafuegos (raster)
    outputs:
       - rodales con cortafuegos (lista de diccionarios con los rodales y sus atributos)
    """
    import os
    from pathlib import Path

    from auxiliary import get_data
    from post_optimization import biomass_with_fire_breacks
    from use_of_QGIS import create_paisaje_con_cortafuegos

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    area_estudio = str(Path("test/data_modificada/proto_mod.shp"))
    cortafuegos = str(Path("cortafuegos/cortafuegos_0.02.tif"))
    create_paisaje_con_cortafuegos(area_estudio, cortafuegos)  # crear paisaje con cortafuegos
    area_con_cortafuegos = Path("cortafuegos/data_cortafuegos/data_modificada/proto_mod.shp")
    gdf_cf = get_data(area_con_cortafuegos)  # se adquiere el shapefile rodales con cortafuegos
    gdf_cf = gdf_cf.sort_values(by="rid")

    # se renombra la columna _mean a prop_cf (proporcion de cortafuegos) mean porque venia de estadistica zonal
    gdf_cf.rename(columns={"_mean": "prop_cf"}, inplace=True)
    rodales_cf = biomass_with_fire_breacks(rodales, gdf_cf, "rid")
    return rodales_cf, gdf_cf  # leer el paisaje con cortafuegos


rodales_cf, gdf_cf = rodales_con_cortafuegos(rodales)


def optimizar_modelo(rodales, rodales_cf):  # paso 5
    """paso 5, optimizar el modelo sin incendios ni cortafuegos y otro sin incendios pero con cortafuegos y crear un csv con las soluciones
    inputs:
       - rodales (lista de diccionarios con los rodales y sus atributos)
       - rodales con cortafuegos (lista de diccionarios con los rodales y sus atributos)
       - archivo de configuracion (config.toml)
       - archivo de configuracion  de optimizacion (config_opti.toml)
    outputs:
       - csv con las soluciones (rodales_sin_cortafuegos.csv y rodales_con_cortafuegos.csv)
    """
    from simulator import print_manejos_possibles, read_toml
    from tactico import generate_random_walk_prices, model_t_cplex

    config = read_toml("config.toml")  # leer el archivo de configuración
    config_opti = read_toml("config_opti.toml")  # leer el archivo de configuración
    politicas = print_manejos_possibles(
        config
    )  # lista con los años en los que se puede ralear y cosechar alguno de los rodales simulados
    prices = generate_random_walk_prices(
        config_opti["opti"]["Price"], config["horizonte"], mu=0.05, sigma=0.1
    )  # genera precios aleatorios

    # optimiza el modelo sin incendios ni cortafuegos y crea un csv con las soluciones
    valores_objetivo, soluciones = model_t_cplex(
        rodales, politicas, prices, "rodales_sin_cortafuegos", config, config_opti
    )

    # optimiza el modelo sinn incendios pero con cortafuegos y crea un csv con las soluciones
    valores_objetivo_cf, soluciones_cf = model_t_cplex(
        rodales_cf, politicas, prices, "rodales_con_cortafuegos", config, config_opti
    )

    return valores_objetivo, valores_objetivo_cf, soluciones, soluciones_cf, prices


valores_objetivo, valores_objetivo_cf, soluciones, soluciones_cf, prices = optimizar_modelo(
    rodales, rodales_cf
)  # paso 5


def quemar_soluciones(rodales, rodales_cf, soluciones, soluciones_cf, gdf, gdf_cf):  # paso 6 y 7
    """paso 6 y 7, crear los archivos de combustibles, quemar las soluciones y obtener burn probability adaptado a periodo
    inputs:
       - rodales (lista de diccionarios con los rodales y sus atributos)
       - rodales con cortafuegos (lista de diccionarios con los rodales y sus atributos)
       - soluciones (lista de soluciones sin cortafuegos)
       - soluciones con cortafuegos (lista de soluciones con cortafuegos)
       - archivo de configuracion (config.toml)
         - archivo de configuracion  de optimizacion (config_opti.toml)
      outputs:
         - burn probaility adaptado a periodo soluciones sin cortafuegos
         - burn probaility adaptado a periodo soluciones con cortafuegos
         - archivos de combustibles sin cortafuegos
         - archivos de combustibles con cortafuegos
    """
    from post_optimization import filtro
    from simulator import read_toml
    from use_of_QGIS import burn_prob_sol, fuels_creation

    config = read_toml("config.toml")  # leer el archivo de configuración
    config_opti = read_toml("config_opti.toml")  # leer el archivo de configuración

    filter = filtro(rodales, soluciones)  # f[soluciones][rodales]
    filtro_cf = filtro(rodales_cf, soluciones_cf)

    fuels_creation(gdf, filter, "./soluciones/data_modificada", config, "rid")  # crea los archivos de combustibles
    fuels_creation(gdf_cf, filtro_cf, "./cortafuegos/soluciones/data_modificada", config, "rid")
    # crea los archivos de combustibles con cortafuegos
    bp_sin_cortafuegos = burn_prob_sol(
        config_opti["opti"]["soluciones"],
        ".tif",
        filter,
        "./soluciones/data_modificada",
        corta_fuegos=False,
        id="rid",
        paisaje="./test/data_modificada/proto_mod.shp",
    )  # calcula la probabilidad de incendio sin cortafuegos
    bp_con_cortafuegos = burn_prob_sol(
        config_opti["opti"]["soluciones"],
        ".tif",
        filtro_cf,
        "./cortafuegos/soluciones/data_modificada",
        corta_fuegos=True,
        id="rid",
        paisaje="./test/data_modificada/proto_mod.shp",
    )
    return filter, filtro_cf, bp_sin_cortafuegos, bp_con_cortafuegos


filter, filtro_cf, bp_sin_cortafuegos, bp_con_cortafuegos = quemar_soluciones(
    rodales, rodales_cf, soluciones, soluciones_cf, gdf, gdf_cf
)  # paso 6 y 7


def ajustar_ganancias(filter, filtro_cf, bp_sin_cortafuegos, bp_con_cortafuegos, prices):
    """paso 8 y 9, ajustar las ganancias por solucion post incendios, viendo las nuevas ganancias
    inputs:
       - filter (filtro de rodales y soluciones sin cortafuegos)
       - filtro_cf (filtro de rodales y soluciones con cortafuegos)
       - bp_sin_cortafuegos (probabilidad de incendio sin cortafuegos)
       - bp_con_cortafuegos (probabilidad de incendio con cortafuegos)
       - precios (lista de precios aleatorios)
       outputs:
       - ganancias por solucion sin cortafuegos
       - ganancias por solucion con cortafuegos"""
    from post_optimization import multiplicar_listas, sumar_por_solucion

    # biomasa vendida post incendios sin cortafuegos
    new_biomass = multiplicar_listas(
        bp_sin_cortafuegos, filter
    )  # multiplica la biomasa por la probabilidad de incendio
    new_biomass_con_cortafuegos = multiplicar_listas(bp_con_cortafuegos, filtro_cf)
    biomass_for_solution, vt_sin_cortafuegos = sumar_por_solucion(new_biomass, prices)  # suma la biomasa por solucion
    biomass_for_solution_con_cortafuegos, vt_con_cortafuegos = sumar_por_solucion(new_biomass_con_cortafuegos, prices)
    print("las ganancias por solucion post simulación de incendios son:")

    print(biomass_for_solution)
    # simulas los incendios"""
    print("las ganancias por solucion post simulación de incendios son (con conrtafuegos):")
    print(biomass_for_solution_con_cortafuegos)

    # Encontrar el mejor valor y su índice en ambas listas
    max_sin_cortafuegos = max(biomass_for_solution)
    max_con_cortafuegos = max(biomass_for_solution_con_cortafuegos)

    indice_sin_cortafuegos = biomass_for_solution.index(max_sin_cortafuegos)
    indice_con_cortafuegos = biomass_for_solution_con_cortafuegos.index(max_con_cortafuegos)

    if max_sin_cortafuegos > max_con_cortafuegos:
        mejor_valor = max_sin_cortafuegos
        origen = "sin cortafuegos"
        indice = indice_sin_cortafuegos
    else:
        mejor_valor = max_con_cortafuegos
        origen = "con cortafuegos"
        indice = indice_con_cortafuegos

    print(f"La mejor solución es la solucion {indice + 1} {origen}, y su valor es {mejor_valor}.")
    return biomass_for_solution, biomass_for_solution_con_cortafuegos, vt_sin_cortafuegos, vt_con_cortafuegos


biomass_for_solution, biomass_for_solution_con_cortafuegos, vt_sin_cortafuegos, vt_con_cortafuegos = ajustar_ganancias(
    filter, filtro_cf, bp_sin_cortafuegos, bp_con_cortafuegos, prices
)  # paso 8 y 9w
