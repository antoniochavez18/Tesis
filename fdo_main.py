#!python
from pathlib import Path

from main import (calcular_sensibilidad_cortafuegos,
                  crear_opciones_cortafuegos, optimizar_modelo,
                  rodales_con_cortafuegos, simular_crecimiento)

ashape = Path("fdo/instance1_growth_attributes.shp")
assert ashape.exists()

# 1
gdf, rodales = simular_crecimiento(area_estudio=ashape, id="fid", mid="growth_mid", outfile="bosque_data.csv")

# 2
crear_opciones_cortafuegos(gdf, rodales)

# 3
cortafuegos = calcular_sensibilidad_cortafuegos()

# 4
rodales_cf, gdf_cf = rodales_con_cortafuegos(rodales, cortafuegos)

# 5
valores_objetivo, valores_objetivo_cf, soluciones, soluciones_cf, prices = optimizar_modelo(rodales, rodales_cf)

# paso 6: se crean los combustibles y se guardan en la carpeta de soluciones
filter, filtro_cf = crear_combustibles(rodales, rodales_cf, soluciones, soluciones_cf, gdf, gdf_cf)

# paso 7: se simula incendios y se adapta el burn probability para acumular informacion de años pasados
bp_sin_cortafuegos, bp_con_cortafuegos = quemar_soluciones(
    rodales, rodales_cf, gdf, gdf_cf, filter, filtro_cf, cortafuegos
)

# paso 8 y 9: se adapta ganancias y se elije 1 de las 10 soluciones
biomass_for_solution, biomass_for_solution_con_cortafuegos, vt_sin_cortafuegos, vt_con_cortafuegos = (
    ajustar_ganancias(filter, filtro_cf, bp_sin_cortafuegos, bp_con_cortafuegos, prices)
)
