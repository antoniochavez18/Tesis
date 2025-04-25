import ast
import pandas as pd
import numpy as np
import sys

if sys.version_info >= (3, 11):
    import tomllib

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
else:
    import toml

    config = toml.load("config.toml")

if sys.version_info >= (3, 11):
    import tomllib

    with open("config_opti.toml", "rb") as f:
        config_opti = tomllib.load(f)
else:
    import toml

    config_opti = toml.load("config_opti.toml")


rng = np.random.default_rng(4)


def filtro(rodales, soluciones):
    """filtra los datos de los rodales dependiendo de las distintas soluciones."""
    # Cargamos los datos de los archivos

    # Parámetros
    RR = len(rodales)  # Cantidad de rodales
    # Lista para almacenar las soluciones
    datos_soluciones = []

    # Iterar sobre cada solución (por columnas, ignorando la primera columna de nombres)
    for s in range(config_opti["opti"]["soluciones"]):
        datos_solucion = {}  # Inicializamos un diccionario por solución

        # Iterar sobre cada rodal
        for r in range(RR):
            # Obtener la política
            pol = soluciones[r][s + 1]
            indice = next(i for i, rr in enumerate(rodales) if rr["rid"] == soluciones[r][0])

            # Inicializamos un diccionario por rodal para almacenar "codigo_kitral" y "vendible"
            rodal_data = {}
            rodal_data = {
                "rid": rodales[indice]["rid"],
                "growth_model_id": rodales[indice]["growth_model_id"],
                "edad_inicial": rodales[indice]["edad_inicial"],
            }

            if pol == 0:
                # Si no hay raleo ni cosecha, usar el primer manejo
                rodal_data["codigo_kitral"] = rodales[indice]["manejos"][0]["codigo_kitral"]
                rodal_data["vendible"] = rodales[indice]["manejos"][0]["vendible"]
                rodal_data["biomass"] = rodales[indice]["manejos"][0]["biomass"]
                rodal_data["eventos"] = rodales[indice]["manejos"][0]["eventos"]
            else:
                # Buscar el manejo que coincida con la política
                for m in range(len(rodales[r]["manejos"])):
                    if (
                        rodales[indice]["manejos"][m]["raleo"] == pol[0]
                        and rodales[indice]["manejos"][m]["cosecha"] == pol[1]
                    ):
                        rodal_data["codigo_kitral"] = rodales[indice]["manejos"][m]["codigo_kitral"]
                        rodal_data["vendible"] = rodales[indice]["manejos"][m]["vendible"]
                        rodal_data["biomass"] = rodales[indice]["manejos"][m]["biomass"]
                        rodal_data["eventos"] = rodales[indice]["manejos"][m]["eventos"]

            # Guardamos el diccionario de datos del rodal en la solución correspondiente
            datos_solucion[r] = rodal_data

        # Añadir la solución a la lista de soluciones
        datos_soluciones.append(datos_solucion)

    return datos_soluciones


def multiplicar_listas(bp, filtro):
    """multiplica las biomasas vendibles por 1 - prob de quema de cada rodal, periodo y solución."""
    soluciones = len(bp)  # Número de soluciones (dimensión s)
    rodales = len(bp[0])  # Número de rodales (dimensión r)
    periodos = len(bp[0][0])  # Número de periodos (dimensión t)

    # Crear una lista vacía para almacenar los resultados de la multiplicación
    resultado = [[[0 for t in range(periodos)] for r in range(rodales)] for s in range(soluciones)]

    # Iterar sobre soluciones, rodales y periodos
    biomasa = [0 for _ in range(periodos)]
    biomasa_cf = [0 for _ in range(periodos)]

    for s in range(soluciones):
        for r in range(rodales):
            for t in range(periodos):
                # Multiplicar el valor correspondiente de cada lista
                resultado[s][r][t] = (1 - bp[s][r][t]) * filtro[s][r]["vendible"][t]

    return resultado


def sumar_por_solucion(ganancias_totales, prices):
    """Suma las ganancias totales por solución post incendios."""
    soluciones = len(ganancias_totales)  # Número de soluciones
    rodales = len(ganancias_totales[0])  # Número de rodales
    periodos = len(ganancias_totales[0][0])  # Número de periodos

    # Crear una lista para almacenar los totales por solución
    total = [0 for _ in range(soluciones)]
    vt_por_solucion = [[] for _ in range(soluciones)]  # Almacena los valores v_t por solución

    # Iterar sobre soluciones, rodales y periodos para sumar
    for s in range(soluciones):
        suma_solucion = 0  # Inicializamos la suma para cada solución
        vt = [0 for _ in range(periodos)]  # Valores v_t para la solución actual
        for t in range(periodos):
            discount_factor = (1 + config_opti["opti"]["tasa"]) ** t
            v_t = sum((ganancias_totales[s][r][t] * prices[t]) / discount_factor for r in range(rodales))
            vt[t] = v_t
            suma_solucion += v_t

        total[s] = suma_solucion  # Guardamos la suma total para la solución s
        vt_por_solucion[s] = vt  # Guardamos los valores v_t para la solución s

    return total, vt_por_solucion


def graficar_vt_por_solucion(vt_por_solucion, dataset_name):
    import matplotlib.pyplot as plt

    periodos = len(vt_por_solucion[0])
    soluciones = len(vt_por_solucion)

    plt.figure(figsize=(10, 6))
    for s in range(soluciones):
        plt.plot(range(periodos), vt_por_solucion[s], marker="o", linestyle="-", label=f"Solución {s + 1}")

    plt.title(f"Valores presentes por período para cada solución post incendios({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("v_t (Ventas ajustadas)")
    plt.legend(title="Soluciones")
    plt.grid(True)
    plt.savefig(f"valores_vt_por_solucion_post_fuego_{dataset_name}.png")
    plt.show()


def base_case(rodales):
    # Cargamos los datos de los archivos
    # Parámetros
    RR = len(rodales)  # Cantidad de rodales

    base_case_data = {}

    for r in range(RR):
        # Inicializamos un diccionario por rodal para almacenar "codigo_kitral" y "vendible"
        rodal_data = {
            "rid": rodales[r]["rid"],
            "growth_model_id": rodales[r]["growth_model_id"],
            "edad_inicial": rodales[r]["edad_inicial"],
            "codigo_kitral": rodales[r]["manejos"][0]["codigo_kitral"],
            "biomass": rodales[r]["manejos"][0]["biomass"],
        }
        # Guardamos el diccionario de datos del rodal en la base_case_data correspondiente
        base_case_data[r] = rodal_data

    return base_case_data


def biomass_with_fire_breacks(rodales, gdf_cf, id="fid"):
    import copy

    # Crear una copia profunda para evitar modificar el original
    rodales2 = copy.deepcopy(rodales)
    periodos = config["horizonte"]

    for r in range(len(rodales)):
        prop_rodal_no_cf = gdf_cf.loc[gdf_cf[id] == rodales[r]["rid"], "prop_cf"].values[0]

        for t in range(periodos):
            for m in range(len(rodales[r]["manejos"])):
                rodales2[r]["manejos"][m]["biomass"][t] = rodales[r]["manejos"][m]["biomass"][t] * (
                    1 - prop_rodal_no_cf
                )
                rodales2[r]["manejos"][m]["vendible"][t] = rodales[r]["manejos"][m]["vendible"][t] * (
                    1 - prop_rodal_no_cf
                )

    return rodales2


def prop_quemada(filtro, filtro_cf, bp, bp_cf, dataset_name):
    soluciones = len(bp)  # Número de soluciones
    rodales = len(bp[0])  # Número de rodales
    periodos = len(bp[0][0])  # Número de periodos

    # Crear listas para almacenar las proporciones de biomasas quemadas por solución y periodo
    prop_biomasa_quemada = [[0 for _ in range(periodos)] for _ in range(soluciones)]
    prop_vendible_quemada = [[0 for _ in range(periodos)] for _ in range(soluciones)]
    prop_biomasa_quemada_cf = [[0 for _ in range(periodos)] for _ in range(soluciones)]
    prop_vendible_quemada_cf = [[0 for _ in range(periodos)] for _ in range(soluciones)]

    # Iterar sobre soluciones, rodales y periodos para calcular la proporción de biomasa quemada
    for s in range(soluciones):
        for t in range(periodos):
            total_biomass = sum(filtro[s][r]["biomass"][t] for r in range(rodales))
            total_vendible = sum(filtro[s][r]["vendible"][t] for r in range(rodales))
            total_biomass_cf = sum(filtro_cf[s][r]["biomass"][t] for r in range(rodales))
            total_vendible_cf = sum(filtro_cf[s][r]["vendible"][t] for r in range(rodales))

            if total_biomass > 0:
                prop_biomasa_quemada[s][t] = (
                    sum((bp[s][r][t] * filtro[s][r]["biomass"][t]) for r in range(rodales)) / total_biomass
                )
            if total_vendible > 0:
                prop_vendible_quemada[s][t] = (
                    sum((bp[s][r][t] * filtro[s][r]["vendible"][t]) for r in range(rodales)) / total_vendible
                )
            if total_biomass_cf > 0:
                prop_biomasa_quemada_cf[s][t] = (
                    sum((bp_cf[s][r][t] * filtro_cf[s][r]["biomass"][t]) for r in range(rodales)) / total_biomass_cf
                )
            if total_vendible_cf > 0:
                prop_vendible_quemada_cf[s][t] = (
                    sum((bp_cf[s][r][t] * filtro_cf[s][r]["vendible"][t]) for r in range(rodales)) / total_vendible_cf
                )

    # Graficar la proporción de biomasa quemada por periodo
    import matplotlib.pyplot as plt

    periodos_range = np.arange(periodos)
    colors = plt.cm.tab10.colors  # Colores de la gama tab10

    plt.figure(figsize=(10, 6))
    for s in range(soluciones):
        plt.plot(
            periodos_range,
            prop_biomasa_quemada[s],
            marker="o",
            linestyle="-",
            label=f"Solución {s + 1} sin CF",
            color=colors[s],
        )
        plt.plot(
            periodos_range,
            prop_biomasa_quemada_cf[s],
            marker="^",
            linestyle="--",
            label=f"Solución {s + 1} con CF",
            color=colors[s],
            markersize=8,
        )

    plt.title(f"Proporción de biomasa quemada por período para cada solución ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Proporción de biomasa quemada")
    plt.legend(title="Soluciones")
    plt.grid(True)
    plt.savefig(f"prop_biomasa_quemada_por_solucion_{dataset_name}.png")
    plt.show()

    # Graficar la proporción de biomasa vendible quemada por periodo
    plt.figure(figsize=(10, 6))
    for s in range(soluciones):
        plt.plot(
            periodos_range,
            prop_vendible_quemada[s],
            marker="o",
            linestyle="-",
            label=f"Solución {s + 1} sin CF",
            color=colors[s],
        )
        plt.plot(
            periodos_range,
            prop_vendible_quemada_cf[s],
            marker="^",
            linestyle="--",
            label=f"Solución {s + 1} con CF",
            color=colors[s],
            markersize=8,
        )

    plt.title(f"Proporción de pérdidas por período para cada solución ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Proporción de biomasa vendible quemada")
    plt.legend(title="Soluciones")
    plt.grid(True)
    plt.savefig(f"prop_vendible_quemada_por_solucion_{dataset_name}.png")
    plt.show()

    return prop_biomasa_quemada, prop_vendible_quemada, prop_biomasa_quemada_cf, prop_vendible_quemada_cf


def biom_quemada(filtro, filtro_cf, bp, bp_cf, dataset_name):
    soluciones = len(bp)  # Número de soluciones
    rodales = len(bp[0])  # Número de rodales
    periodos = len(bp[0][0])  # Número de periodos

    # Crear listas para almacenar las biomasas quemadas por solución y periodo
    biomasa_quemada = [[0 for _ in range(periodos)] for _ in range(soluciones)]
    vendible_quemada = [[0 for _ in range(periodos)] for _ in range(soluciones)]
    biomasa_quemada_cf = [[0 for _ in range(periodos)] for _ in range(soluciones)]
    vendible_quemada_cf = [[0 for _ in range(periodos)] for _ in range(soluciones)]

    # Iterar sobre soluciones, rodales y periodos para calcular la biomasa quemada
    for s in range(soluciones):
        for t in range(periodos):
            biomasa_quemada[s][t] = sum((bp[s][r][t] * filtro[s][r]["biomass"][t]) for r in range(rodales))
            biomasa_quemada_cf[s][t] = sum((bp_cf[s][r][t] * filtro_cf[s][r]["biomass"][t]) for r in range(rodales))
            vendible_quemada[s][t] = sum((bp[s][r][t] * filtro[s][r]["vendible"][t]) for r in range(rodales))
            vendible_quemada_cf[s][t] = sum((bp_cf[s][r][t] * filtro_cf[s][r]["vendible"][t]) for r in range(rodales))

    biomasa_total = sum(biomasa_quemada[0][t] for t in range(periodos))
    vendible_total = sum(vendible_quemada[0][t] for t in range(periodos))
    biomasa_total_cf = sum(biomasa_quemada_cf[1][t] for t in range(periodos))
    vendible_total_cf = sum(vendible_quemada_cf[1][t] for t in range(periodos))

    # Graficar la biomasa quemada por periodo
    import matplotlib.pyplot as plt

    periodos_range = np.arange(periodos)
    colors = plt.cm.tab10.colors  # Colores de la gama tab10

    plt.figure(figsize=(10, 6))
    for s in range(soluciones):
        plt.plot(
            periodos_range,
            biomasa_quemada[s],
            marker="o",
            linestyle="-",
            label=f"Solución {s + 1} sin CF",
            color=colors[s],
        )
        plt.plot(
            periodos_range,
            biomasa_quemada_cf[s],
            marker="^",
            linestyle="--",
            label=f"Solución {s + 1} con CF",
            color=colors[s],
            markersize=8,
        )

    plt.title(f"Biomasa quemada por período para cada solución ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa quemada")
    plt.legend(title="Soluciones")
    plt.grid(True)
    plt.savefig(f"biomasa_quemada_por_solucion_{dataset_name}.png")
    plt.show()

    # Graficar la biomasa vendible quemada por periodo
    plt.figure(figsize=(10, 6))
    for s in range(soluciones):
        plt.plot(
            periodos_range,
            vendible_quemada[s],
            marker="o",
            linestyle="-",
            label=f"Solución {s + 1} sin CF",
            color=colors[s],
        )
        plt.plot(
            periodos_range,
            vendible_quemada_cf[s],
            marker="^",
            linestyle="--",
            label=f"Solución {s + 1} con CF",
            color=colors[s],
            markersize=8,
        )

    plt.title(f"Perdidas por período para cada solución ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa vendible quemada")
    plt.legend(title="Soluciones")
    plt.grid(True)
    plt.savefig(f"vendible_quemada_por_solucion_{dataset_name}.png")
    plt.show()

    # Graficar la biomasa por periodo comparando la solución 1 sin cortafuegos y la solución 3 con cortafuegos
    plt.figure(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    plt.bar(periodos_range - width / 2, biomasa_quemada[0], width=width, label="Mejor Solución sin CF", color=colors[0])
    plt.bar(
        periodos_range + width / 2, biomasa_quemada_cf[1], width=width, label="Mejor Solución con CF", color=colors[1]
    )

    # Agregar segunda "leyenda" con valores específicos
    texto_explicativo = (
        f"Biomasa total quemada: {biomasa_total:.2f}\n" f"Biomasa total con cortafuegos: {biomasa_total_cf:.2f}"
    )
    plt.text(
        0.5,
        0.85,
        texto_explicativo,
        transform=plt.gca().transAxes,  # Ubicación relativa en la esquina superior derecha
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.title(f"Comparación de biomasa quemada por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa quemada")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"comparacion_biomasa_quemada_{dataset_name}.png")
    plt.show()

    # Graficar la biomasa vendible por periodo comparando la solución 1 sin cortafuegos y la solución 3 con cortafuegos
    plt.figure(figsize=(10, 6))
    plt.bar(
        periodos_range - width / 2, vendible_quemada[0], width=width, label="Mejor Solución sin CF", color=colors[0]
    )
    plt.bar(
        periodos_range + width / 2, vendible_quemada_cf[1], width=width, label="Mejor Solución con CF", color=colors[1]
    )

    # Agregar segunda "leyenda" con valores específicos
    texto_explicativo = (
        f"Biomasa total quemada: {vendible_total:.2f}\n" f"Biomasa total con cortafuegos: {vendible_total_cf:.2f}"
    )
    plt.text(
        0.95,
        0.85,
        texto_explicativo,
        transform=plt.gca().transAxes,  # Ubicación relativa en la esquina superior derecha
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.title(f"Comparación perdidas por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa vendible quemada")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"comparacion_vendible_quemada_{dataset_name}.png")
    plt.show()

    return biomasa_quemada, vendible_quemada, biomasa_quemada_cf, vendible_quemada_cf


def biom_final(filtro, bp):
    soluciones = len(bp)  # Número de soluciones
    rodales = len(bp[0])  # Número de rodales

    # Crear una lista para almacenar las proporciones por solución y periodo
    biomass_for_solution = [0 for _ in range(soluciones)]
    biomass_for_solution_no_quema = [0 for _ in range(soluciones)]
    # Iterar sobre soluciones, rodales y periodos para calcular la proporción quemada
    for s in range(soluciones):
        suma_solucion = sum(((1 - bp[s][r][-1]) * filtro[s][r]["biomass"][-1]) for r in range(rodales))
        suma_solucion_no_quema = sum((filtro[s][r]["biomass"][-1]) for r in range(rodales))
        biomass_for_solution[s] = suma_solucion
        biomass_for_solution_no_quema[s] = suma_solucion_no_quema
    return biomass_for_solution, biomass_for_solution_no_quema


# Tu lista
# Abre (o crea) un archivo de texto en modo escritura
"""with open("bp_sin_cortafuegos.txt", "w") as archivo:
    for item in bp_sin_cortafuegos:
        archivo.write("%s\n" % item)


with open("bp_con_cortafuegos.txt", "w") as archivo:
    for item in bp_con_cortafuegos:
        archivo.write("%s\n" % item)"""


def grafico_ahora_si(filtro, filtro_cf, bp, bp_cf, dataset_name):
    rodales = len(bp[0])  # Número de rodales
    periodos = len(bp[0][0])  # Número de periodos
    biomasa_quemada = [0 for _ in range(periodos)]
    vendible_quemada = [0 for _ in range(periodos)]
    biomasa_quemada_cf = [0 for _ in range(periodos)]
    vendible_quemada_cf = [0 for _ in range(periodos)]
    biomasa = [0 for _ in range(periodos)]
    vendible = [0 for _ in range(periodos)]
    biomasa_cf = [0 for _ in range(periodos)]
    vendible_cf = [0 for _ in range(periodos)]
    resto = [0 for _ in range(periodos)]
    resto_cf = [0 for _ in range(periodos)]
    prop_quemada = [0 for _ in range(periodos)]
    prop_quemada_cf = [0 for _ in range(periodos)]
    prop_vendible_quemada = [0 for _ in range(periodos)]
    prop_vendible_quemada_cf = [0 for _ in range(periodos)]
    prop_vendible = [0 for _ in range(periodos)]
    prop_vendible_cf = [0 for _ in range(periodos)]
    prop_resto = [0 for _ in range(periodos)]
    prop_resto_cf = [0 for _ in range(periodos)]

    for t in range(periodos):
        biomasa_quemada[t] = sum((bp[0][r][t] * filtro[0][r]["biomass"][t]) for r in range(rodales))
        vendible_quemada[t] = sum((bp[0][r][t] * filtro[0][r]["vendible"][t]) for r in range(rodales))
        biomasa_quemada_cf[t] = sum((bp_cf[1][r][t] * filtro_cf[1][r]["biomass"][t]) for r in range(rodales))
        vendible_quemada_cf[t] = sum((bp_cf[1][r][t] * filtro_cf[1][r]["vendible"][t]) for r in range(rodales))
        biomasa[t] = sum((filtro[0][r]["biomass"][t]) for r in range(rodales))
        vendible[t] = sum(((1 - bp[0][r][t]) * filtro[0][r]["vendible"][t]) for r in range(rodales))
        biomasa_cf[t] = sum((filtro_cf[1][r]["biomass"][t]) for r in range(rodales))
        vendible_cf[t] = sum(((1 - bp_cf[1][r][t]) * filtro_cf[1][r]["vendible"][t]) for r in range(rodales))
        resto[t] = biomasa[t] - biomasa_quemada[t] - vendible[t]
        resto_cf[t] = biomasa_cf[t] - biomasa_quemada_cf[t] - vendible_cf[t]
        prop_quemada[t] = biomasa_quemada[t] / biomasa[t] if biomasa[t] > 0 else 0
        prop_quemada_cf[t] = biomasa_quemada_cf[t] / biomasa_cf[t] if biomasa_cf[t] > 0 else 0
        prop_vendible_quemada[t] = vendible_quemada[t] / (vendible[t] + vendible_quemada[t]) if vendible[t] > 0 else 0
        prop_vendible_quemada_cf[t] = (
            vendible_quemada_cf[t] / (vendible_cf[t] + vendible_quemada_cf[t]) if vendible_cf[t] > 0 else 0
        )
        prop_vendible[t] = vendible[t] / biomasa[t] if biomasa[t] > 0 else 0
        prop_vendible_cf[t] = vendible_cf[t] / biomasa_cf[t] if biomasa_cf[t] > 0 else 0
        prop_resto[t] = resto[t] / biomasa[t] if biomasa[t] > 0 else 0
        prop_resto_cf[t] = resto_cf[t] / biomasa_cf[t] if biomasa_cf[t] > 0 else 0

    biomasa_quema_total = sum(biomasa_quemada[t] for t in range(periodos))
    vendible_quema_total = sum(vendible_quemada[t] for t in range(periodos))
    biomasa_total_quema_cf = sum(biomasa_quemada_cf[t] for t in range(periodos))
    vendible_total_quema_cf = sum(vendible_quemada_cf[t] for t in range(periodos))
    vendible_total = sum(vendible[t] for t in range(periodos))
    vendible_total_cf = sum(vendible_cf[t] for t in range(periodos))

    # Graficar la biomasa quemada por periodo
    import matplotlib.pyplot as plt

    periodos_range = np.arange(periodos)
    # Graficar la biomasa quemada por periodo en un gráfico de barras
    plt.figure(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    plt.bar(periodos_range - width / 2, biomasa_quemada, width=width, label="Biomasa quemada sin CF")
    plt.bar(periodos_range + width / 2, biomasa_quemada_cf, width=width, label="Biomasa quemada con CF")
    plt.title(f"Comparación de biomasa quemada por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa quemada")
    plt.legend()
    plt.grid(True)
    texto_explicativo = (
        f"Biomasa quemada total sin CF: {biomasa_quema_total:.2f}\n"
        f"Biomasa quemada total con CF: {biomasa_total_quema_cf:.2f}"
    )
    plt.text(
        0.5,
        0.85,
        texto_explicativo,
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="right",
    )
    plt.savefig(f"biomasa_quemada_{dataset_name}.png")
    plt.show()

    # Graficar la  prop biomasa quemada por periodo en un gráfico de barras
    plt.figure(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    plt.bar(periodos_range - width / 2, prop_quemada, width=width, label="Proporción biomasa quemada sin CF")
    plt.bar(periodos_range + width / 2, prop_quemada_cf, width=width, label="Proporción biomasa quemada con CF")
    plt.title(f"Comparación de proporción biomasa quemada por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Proporción biomasa quemada")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Proporción_biomasa_quemada_{dataset_name}.png")
    plt.show()

    # Graficar la biomasa vendible quemada por periodo en un gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(periodos_range - width / 2, vendible_quemada, width=width, label="Biomasa vendible quemada sin CF")
    plt.bar(periodos_range + width / 2, vendible_quemada_cf, width=width, label="Biomasa vendible quemada con CF")
    plt.title(f"Comparación de Pérdidas por biomasa vendible quemada por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa vendible quemada")
    plt.legend()
    plt.grid(True)
    texto_explicativo = (
        f"Biomasa vendible quemada total sin CF: {vendible_quema_total:.2f}\n"
        f"Biomasa vendible quemada total con CF: {vendible_total_quema_cf:.2f}"
    )
    plt.text(
        0.98,
        0.85,
        texto_explicativo,
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="right",
    )
    plt.savefig(f"vendible_quemada_{dataset_name}.png")
    plt.show()

    # Graficar la  prop biomasa vendible quemada por periodo en un gráfico de barras
    plt.figure(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    plt.bar(periodos_range - width / 2, prop_vendible_quemada, width=width, label="Proporción perdidas sin CF")
    plt.bar(periodos_range + width / 2, prop_vendible_quemada_cf, width=width, label="Proporción pérdidas con CF")
    plt.title(f"Comparación proporción pérdidas por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Proporción pérdidas")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Proporción_biomasa_vendible_quemada_{dataset_name}.png")
    plt.show()

    # Graficar la biomasa vendible quemada por periodo en un gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(periodos_range - width / 2, vendible, width=width, label="Biomasa vendible sin CF")
    plt.bar(periodos_range + width / 2, vendible_cf, width=width, label="Biomasa vendible con CF")
    plt.title(f"Comparación de biomasa vendible por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa vendible")
    plt.legend()
    plt.grid(True)
    texto_explicativo = (
        f"Biomasa vendible total sin CF: {vendible_total:.2f}\n"
        f"Biomasa vendible total con CF: {vendible_total_cf:.2f}"
    )
    plt.text(
        0.95,
        0.85,
        texto_explicativo,
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="right",
    )
    plt.savefig(f"vendible_{dataset_name}.png")
    plt.show()

    # Graficar la  prop biomasa vendida por periodo con respecto a biomasa total en un gráfico de barras
    plt.figure(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    plt.bar(periodos_range - width / 2, prop_vendible, width=width, label="Proporción biomasa vendida sin CF")
    plt.bar(periodos_range + width / 2, prop_vendible_cf, width=width, label="Proporción biomasa vendida con CF")
    plt.title(f"Comparación proporción biomasa vendida por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Proporción vendida")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Proporción_biomasa_vendible_{dataset_name}.png")
    plt.show()

    # Graficar la biomasa restante por periodo en un gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(periodos_range - width / 2, resto, width=width, label="Biomasa restante sin CF")
    plt.bar(periodos_range + width / 2, resto_cf, width=width, label="Biomasa restante con CF")
    plt.title(f"Comparación de biomasa restante por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Biomasa restante")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"resto_{dataset_name}.png")
    plt.show()

    # Graficar la  prop biomasa restante por periodo con respecto a biomasa total en un gráfico de barras
    plt.figure(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    plt.bar(periodos_range - width / 2, prop_resto, width=width, label="Proporción biomasa restante sin CF")
    plt.bar(periodos_range + width / 2, prop_resto_cf, width=width, label="Proporción biomasa restante con CF")
    plt.title(f"Comparación proporción biomasa restante por período ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("Proporción biomasa restante")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Proporción_biomasa_resto_{dataset_name}.png")
    plt.show()

    return (
        biomasa_quemada,
        vendible_quemada,
        biomasa_quemada_cf,
        vendible_quemada_cf,
        biomasa,
        vendible,
        biomasa_cf,
        vendible_cf,
        resto,
        resto_cf,
    )


"""(biomasa_quema,vendible_quema,biomasa_quema_cf,vendible_quema_cf,biomasa,vendible,biomasa_cf,vendible_cf,resto,resto_cf) = grafico_ahora_si(f, f2, bp_sin_cortafuegos, bp_con_cortafuegos, "mejor solucion sin y con cortafuegos")
"""
#  1581 biomasa quitada con cortafuegos
#  81197 biomasa total sin cortafuegos
