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
calcular_sensibilidad_cortafuegos()

# 4
rodales_cf, gdf_cf = rodales_con_cortafuegos(rodales)

# 5
valores_objetivo, valores_objetivo_cf, soluciones, soluciones_cf, prices = optimizar_modelo(rodales, rodales_cf)
