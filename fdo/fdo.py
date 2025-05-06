# python3
"""
hey

# TODO
kitral_cod
"""
# 1. subdivide
"""
select feature
 $area > 50000

native:subdivide
{ 'INPUT' : '/home/fdo/source/fire/antotesis/fdo/instance.gpkg|layername=catastro_proto2|geometrytype=MultiPolygon|uniqueGeometryType=yes', 'MAX_NODES' : 256, 'OUTPUT' : 'TEMPORARY_OUTPUT' }

OUTPUT: subdivided.gpkg
"""
# 2. Multipart to singlepart
"""
native:multiparttosinlgepart
{ 'INPUT' : '/home/fdo/source/fire/antotesis/fdo/subdivided.gpkg|layername=subdivided', 'OUTPUT' : 'TEMPORARY_OUTPUT' }

OUTPUT: single_parts.gpkg
"""
# 3. Merge small plantations <=1 ha
"""
# atribute table
$area < 10000 and combustible >= 19 and combustible <= 28

qgis:eliminateselectedpolygons
{ 'INPUT' : '/home/fdo/source/fire/antotesis/fdo/single_parts.gpkg|layername=single_parts', 'MODE' : 0, 'OUTPUT' : 'TEMPORARY_OUTPUT' }

OUTPUT: eliminated.gpkg
"""
# 3.z detour: view gpkg filter with sql
"""
SELECT *
FROM single_parts
WHERE ST_Area(geom) < 10000 and "combustible" IN (19, 20, 21, 22, 23, 24, 25,26, 27, 28);
"""
# 4. Create attributes for growth model
"""

"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from simulator import obtener_datos_kitral

# modelos
tabla = pd.read_csv("tabla.csv")

# instancia
ashape = Path("fdo/instance.shp")
assert ashape.exists()

gdf = gpd.read_file(ashape)
print(gdf.shape)

gdf[["error", "Especie", "EdadRango", "Condicion"]] = gdf["combustibl"].apply(obtener_datos_kitral).apply(pd.Series)
print(gdf.shape)


# ver el mapa del paper de crecimiento para ubicar la zona
def set_zona(x):
    if x == "eucalyptus":
        return "Z1"
    elif x == "pino":
        return "Z6"
    return np.nan


gdf["Zona"] = gdf["Especie"].apply(set_zona)
print(gdf.shape)

gdf[["Zona", "Especie", "EdadRango", "Condicion", "error"]].describe()

# edad sorteada
gdf["edad"] = gdf.loc[~pd.isna(gdf["EdadRango"]), "EdadRango"].apply(lambda x: np.random.randint(*x))
# np.unique(gdf["edad"], return_counts=True)


def set_growth_model_id(x):
    # x = gdf[gdf["Especie"] == "eucalyptus"].iloc[0]
    # x = gdf[gdf["Especie"] == "pino"].iloc[0]
    opciones = tabla[
        (tabla["Especie"] == x["Especie"]) & (tabla["Zona"] == x["Zona"]) & (tabla["Condicion"] == x["Condicion"])
    ]
    if opciones.empty:
        return np.nan
    else:
        draw_id = np.random.choice(opciones.id)
        return draw_id


gdf["growth_mid"] = gdf.loc[~pd.isna(gdf["Especie"])].apply(set_growth_model_id, axis=1)
#
gdf.rename(columns={"combustibl": "kitral_cod"}, inplace=True)

gdf.to_file("fdo/instance1_growth_attributes.shp", driver="ESRI Shapefile")
