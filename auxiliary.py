#!/usr/bin/env python
import numpy as np

models = np.genfromtxt(
    "tabla.csv",
    delimiter=",",
    names=True,  # id,next,Especie,Zona,DensidadInicial,SiteIndex,Manejo,Condicion,α,β,γ
    dtype=None,
    encoding="utf-8",
)


def get_data(filepath=".\\test\\data_base\\proto.shp"):
    import geopandas as gpd

    """Read data of forest from a shp"""
    gdf = gpd.read_file(filepath)
    return gdf
    """from osgeo import ogr

    # Ruta al archivo shapefile
    ruta_shapefile = 'ruta/al/archivo.shp'

    # Abrir el shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(ruta_shapefile, 0)  # 0 significa modo de solo lectura
    if not datasource:
        print("No se pudo abrir el archivo shapefile.")
        exit()

    # Obtener la capa (layer)
    layer = datasource.GetLayer()

    # Iterar sobre los features en la capa
    for feature in layer:
        # Obtener la geometría del feature
        geom = feature.GetGeometryRef()

        # Listar los atributos del feature
        for field in feature.keys():
            print(f"{field}: {feature.GetField(field)}")

        # Imprimir la geometría como WKT (Well-Known Text)
        print(geom.ExportToWkt())
        print('----------')

    # Cerrar el shapefile
    datasource = None"""


def create_forest(gdf, id="fid",outfile="bosque_data.csv"):
    data_rodales = gdf.dropna(subset=["edad"])
    data_rodales_2 = data_rodales.loc[data_rodales["area_ha"] > 0]
    bos_names = ["rid", "mid", "edad_inicial", "ha"]  # aprender hacer formato decente
    rodales = []

    for idx, r in data_rodales_2.iterrows():
        # model = rng.choice(models)
        # print(model)
        e0 = r["edad"]
        ha = r["area_ha"]
        rodal = {
            "rid": r[id],  # r["fid"] en caso habitual
            "mid": r["id"],
            "edad_inicial": e0,
            "ha": ha,
        }
        rodales += [rodal]

    bos = np.array(
        [tuple(r[k] for k in bos_names) for r in rodales],
        dtype=[("rid", "i4"), ("mid", "i4"), ("edad_inicial", "i4"), ("ha", "f4")],
    )
    np.savetxt(
        outfile, bos, delimiter=",", header=",".join(bos_names), comments="", fmt=["%d", "%d", "%d", "%.2f"]
    )


def plot_1_id_model(horizon: int = 40, show=True, save=False, target_id: int = 30):
    """Crea gráfico con los cálculos de biomasa por cada id del árbol eje x igual al año y eje y igual a la biomasa"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title("Modelos de crecimiento")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Biomasa Total")

    for model in models:
        if target_id is not None and model["id"] != target_id:
            continue

        x = np.linspace(0, horizon, 1000)  # Ajuste de resolución
        y = model["α"] * x ** model["β"] + model["γ"]

        y_zero_adjusted = np.where(
            x < np.ceil(model["stable_year"]),
            (model["α"] * np.ceil(model["stable_year"]) ** model["β"] + model["γ"]) * x / np.ceil(model["stable_year"]),
            model["α"] * x ** model["β"] + model["γ"],
        )

        ax.plot(x, y, label="Sin Arreglo", color="blue")
        ax.plot(x, y_zero_adjusted, label="Con Arreglo", color="orange")

        zero = model["stable_year"]

        # Mostrar la línea vertical para 'el año estable'
        ax.axvline(
            x=zero,
            color="r",
            linestyle="--",
            label="Año Estable" if "Año Estable" not in [l.get_label() for l in ax.get_lines()] else "",
        )

        x_integers = np.arange(0, horizon + 1, 1)
        y_integers = model["α"] * x_integers ** model["β"] + model["γ"]
        y_zero_adjusted_integers = np.where(
            x_integers < np.ceil(model["stable_year"]),
            (model["α"] * np.ceil(model["stable_year"]) ** model["β"] + model["γ"])
            * x_integers
            / np.ceil(model["stable_year"]),
            model["α"] * x_integers ** model["β"] + model["γ"],
        )

        ax.plot(
            x_integers,
            y_integers,
            "o",
            color="blue",
            label=(
                "Puntos enteros Sin Arreglo"
                if "Puntos enteros Sin Arreglo" not in [l.get_label() for l in ax.get_lines()]
                else ""
            ),
        )
        ax.plot(
            x_integers,
            y_zero_adjusted_integers,
            "o",
            color="orange",
            label=(
                "Puntos enteros Con Arreglo"
                if "Puntos enteros Con Arreglo" not in [l.get_label() for l in ax.get_lines()]
                else ""
            ),
        )

    ax.axhline(0, color="black", linestyle="--")
    ax.legend()

    # Añadir la leyenda explicativa debajo del gráfico
    plt.figtext(
        0.5,
        -0.15,
        "La línea azul representa el modelo de crecimiento sin ajuste.\nLa línea naranja muestra el modelo ajustado para valores menores al 'zero'.\nLa línea roja punteada indica el valor 'zero'.",
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )

    if save:
        plt.savefig(f"model_{target_id}_id.png")
    if show:
        plt.show()


def solve_numeric():
    """resolver numericamente los zeros de cada ecuación de crecimiento"""
    from scipy.optimize import fsolve

    zeros = []
    for model in models:
        zeros += [fsolve(lambda x: model["α"] * x ** model["β"] + model["γ"], config["horizonte"])[0]]
    display(f"numeric_{zeros=}")
    return zeros


def solve_symbolic():
    """calcula el zeros simbolico de la ecuacion de biomasa"""
    from sympy import solve, symbols

    x, α, β, γ = symbols("x α β γ")
    display(solve(α * x**β + γ, x))
    # [(-γ/α)**(1/β)]
    zeros = []
    for model in models:
        zeros += [(-model["γ"] / model["α"]) ** (1 / model["β"])]
    display(f"symbolic_{zeros=}")
    return zeros


def append_zeros():
    """agregar zeros a los modelos"""
    global models
    import numpy.lib.recfunctions as rfn

    for name, zeros in zip(["zero_numeric", "zero_symbolic"], [solve_numeric(), solve_symbolic()]):
        models = rfn.append_fields(models, name, zeros, usemask=False)

    zn = models["zero_numeric"]
    zs = models["zero_symbolic"]
    mask = ~np.isnan(zn) & ~np.isnan(zs)
    assert np.isclose(zn[mask], zs[mask]).all()

    names = ",".join(models.dtype.names)
    np.savetxt("new_table.csv", models, delimiter=",", header=names, comments="", fmt="%s")


def superpro():
    import pickle

    # store
    with open("all.pickle", "wb") as f:
        pickle.dump((config, models, rodales), f)

    # pickup
    with open("all.pickle", "rb") as f:
        config, models, rodales = pickle.load(f)
