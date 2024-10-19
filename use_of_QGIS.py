#!python3
"""
1. Enables a python script to run the QGIS environment,
without launching the QGIS GUI.
2. Enables a user located processing plugin to be loaded

Programmers must:
1. Adjust the path to the QGIS installation 
(on windows also adjust qgis versions)

2. load the QGIS python environment to run

Helpful commands:
call "%USERPROFILE%\source\pyenv\python-qgis-cmd.bat"
%autoindent
ipython

References:
https://fire2a.github.io/docs/docs/qgis/README.html
https://gis.stackexchange.com/a/408738
https://gis.stackexchange.com/a/172849
"""
import sys
from platform import system as platform_system
from shutil import which
from os import pathsep, environ
from fire2a.raster import read_raster_band

from qgis.core import QgsApplication, QgsRasterLayer
import numpy as np


if sys.version_info >= (3, 11):
    import tomllib

    with open("config_test.toml", "rb") as f:
        config = tomllib.load(f)
else:
    import toml

    config = toml.load("config.toml")

#
## PART 1
#
if platform_system() == "Windows":
    QgsApplication.setPrefixPath("C:\\PROGRA~1\\QGIS33~1.2", True)
else:
    QgsApplication.setPrefixPath("/usr", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Append the path where processing plugin can be found
if platform_system() == "Windows":
    sys.path.append("C:\\PROGRA~1\\QGIS33~1.2\\apps\\qgis\\python\\plugins")
else:
    sys.path.append("/usr/share/qgis/python/plugins")

import processing
from processing.core.Processing import Processing

Processing.initialize()

if platform_system() == "Windows":
    sys.path.append("C:\\Users\\xunxo\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins")
else:
    sys.path.append("/home/fdo/.local/share/QGIS/QGIS3/profiles/default/python/plugins/")
# Add the algorithm provider
from fireanalyticstoolbox.fireanalyticstoolbox_provider import FireToolboxProvider

provider = FireToolboxProvider()
QgsApplication.processingRegistry().addProvider(provider)

print(processing.algorithmHelp("fire2a:scar"))


def burn_prob(path):
    result = processing.run(
        "fire2a:cell2firesimulator",
        {
            "CbdRaster": None,
            "CbhRaster": None,
            "CcfRaster": None,
            "DryRun": False,
            "ElevationRaster": None,
            "EnableCrownFire": False,
            "FireBreaksRaster": None,
            "FoliarMoistureContent": 66,
            "FuelModel": 1,
            "FuelRaster": path,
            "IgnitionMode": 0,
            "IgnitionPointVectorLayer": None,
            "IgnitionProbabilityMap": None,
            "IgnitionRadius": 0,
            "InstanceDirectory": "TEMPORARY_OUTPUT",
            "InstanceInProject": False,
            "LiveAndDeadFuelMoistureContentScenario": 2,
            "NumberOfSimulations": 10,
            "OtherCliArgs": "",
            "OutputOptions": [1, 2, 3, 4],
            "RandomNumberGeneratorSeed": 123,
            "ResultsDirectory": "TEMPORARY_OUTPUT",
            "ResultsInInstance": True,
            "SetFuelLayerStyle": False,
            "SimulationThreads": 15,
            "WeatherDirectory": "",
            "WeatherFile": "C:\\Users\\xunxo\\OneDrive\\Escritorio\\Kitral\\Portezuelo\\Weather.csv",
            "WeatherMode": 0,
        },
    )
    if rd := result.get("ResultsDirectory"):
        pass
    else:
        print("eerrrrr")

    bundle = processing.run(
        "fire2a:simulationresultsprocessing",
        {
            "BaseLayer": "C:\\Users\\xunxo\\Documents\\fire_soluciones\\asas.tif",
            "EnablePropagationDiGraph": False,
            "EnablePropagationScars": False,
            "OutputDirectory": "TEMPORARY_OUTPUT",
            "ResultsDirectory": result["ResultsDirectory"],
        },
    )

    data, _, _ = read_raster_band(bundle["BurnProbability"], 1)
    a = np.concatenate((data[0], data[1], data[2], data[3], data[4], data[5]))
    return a


def burn_prob_sol(num_soluciones):
    periodos = config["horizonte"]
    bp = []

    # Iterar sobre todas las soluciones
    for s in range(num_soluciones):
        solucion_bp = []
        # Iterar sobre todos los periodos para la solución actual
        for t in range(periodos):
            base_path = f"C:\\Local\\fire_git\\growth_2\\soluciones\\fuels_solucion_{s}_periodo_{t}.asc"
            path_p = base_path.format(s, t)
            solucion_bp.append(burn_prob(path_p))  # Obtener la lista de 36 elementos para este periodo

        # Reorganizar para que tengamos 36 listas de 'n' elementos (uno por cada periodo, por solución)
        reorganizado = list(map(list, zip(*solucion_bp)))

        # Ahora calculamos el promedio de cada valor con todos los periodos anteriores para esta solución
        promedios_solucion = []
        for fila in reorganizado:
            promedios_fila = []
            suma_acumulada = 0  # Para llevar el seguimiento de la suma de valores
            for i in range(len(fila)):
                suma_acumulada += fila[i]  # Actualizamos la suma acumulada
                promedio = suma_acumulada / (i + 1)  # Calculamos el promedio hasta el periodo actual
                promedios_fila.append(promedio)

            promedios_solucion.append(promedios_fila)

        # Guardar los promedios de la solución actual en la lista final
        bp.append(promedios_solucion)

    return bp  # Devuelve bp[s][r][t] donde s es la solución, r es el rodal y t el periodo


"""bp = processing.run(
    "fire2a:burnprobabilitypropagationmetric",
    {
        "authid": "EPSG 32718",
        "BaseLayer": "C:/Users/xunxo/OneDrive/Escritorio/Kitral/Portezuelo/soluciones/fuels_0_periodo_0.asc",
        "BurnProbability": "TEMPORARY_OUTPUT",
        "SampleScarFile": "C:/Users/xunxo/OneDrive/Escritorio/results/Grids/Grids[0-9]*/ForestGrid[0-9]*.csv",
    },
)"""
