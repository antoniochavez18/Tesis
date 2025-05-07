#!python
"""
export QT_QPA_PLATFORM="offscreen"
export CPLEX_STUDIO_DIR2211=/opt/ibm/ILOG/CPLEX_Studio2212
export LD_LIBRARY_PATH=/opt/ibm/ILOG/CPLEX_Studio2212/cplex/bin/x86-64_linux
LD_LIBRARY_PATH
echo LD_LIBRARY_PATH
LD_LIBRARY_PATH="/nuevo/path:$LD_LIBRARY_PATH"
"""

from pathlib import Path

from main import (
    ajustar_ganancias,
    crear_combustibles,
    optimizar_modelo,
    quemar_soluciones,
    rodales_con_cortafuegos,
    simular_crecimiento,
)

ashape = Path("fdo/instance1_growth_attributes.shp")
assert ashape.exists()

# 1
gdf, rodales = simular_crecimiento(area_estudio=ashape, id="fid", mid="growth_mid", outfile="bosque_data.csv")

# 2
# crear_opciones_cortafuegos(gdf, rodales)

# 3
# cortafuegos = calcular_sensibilidad_cortafuegos()
cortafuegos = str(Path("cortafuegos/cortafuegos_0.01.tif"))
# 4
rodales_cf, gdf_cf = rodales_con_cortafuegos(
    rodales,
    cortafuegos,
    area_estudio=ashape,
    area_con_cortafuegos=str(Path("cortafuegos/data_cortafuegos/data_modificada/proto_mod.shp")),
)

# 5
valores_objetivo, valores_objetivo_cf, soluciones, soluciones_cf, prices = optimizar_modelo(rodales, rodales_cf)

# paso 6: se crean los combustibles y se guardan en la carpeta de soluciones
filter, filtro_cf = crear_combustibles(rodales, rodales_cf, soluciones, soluciones_cf, gdf, gdf_cf)

# paso 7: se simula incendios y se adapta el burn probability para acumular informacion de años pasados
bp_sin_cortafuegos, bp_con_cortafuegos = quemar_soluciones(
    rodales, rodales_cf, gdf, gdf_cf, filter, filtro_cf, cortafuegos, str(ashape)
)

# paso 8 y 9: se adapta ganancias y se elije 1 de las 10 soluciones
biomass_for_solution, biomass_for_solution_con_cortafuegos, vt_sin_cortafuegos, vt_con_cortafuegos = ajustar_ganancias(
    filter, filtro_cf, bp_sin_cortafuegos, bp_con_cortafuegos, prices
)
