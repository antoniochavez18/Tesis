#!/usr/bin/env python
"""
Simulador de crecimiento de rodales con distintos planes de manejo, crecimiento de acuerdo a modelos de crecimiento (tabla.csv)

Una ejecucion genera un lista de rodales (:dict) con sus manejos (:dict), biomasa, eventos, biomasa vendible y codigos de kitral fuel model

    rodales = [ ...
        {'rid': 9,           # rodal id
         'mid': 24,          # model id
         'edad_inicial': 17,
         'edad_final': 27,
         'ha': 14,
         'manejos' : [ ...
            {'rid': 9,
             'cosecha': 18,
             'raleo': 6,
             'biomass': array([2593., 0., 54.7, 76.9, 118.8, 182.7, 270.2, 40., 46.7, 53.4]),
             'edades': array([17,  0,  1,  2,  3,  4,  5,  6,  7,  8]),
             'eventos': ['', 'c', '', '', '', '', '', 'r', '', ''],
             'vendible': array([0. , 2868.1, 0., 0., 0., 0., 0., 342.6, 0., 0.]),
             'codigo_kitral': [24, 25, 19, 19, 19, 20, 20, 23, 23, 23]
            }
         ]
        }
    ]

Uso:
    - crear/editar config.toml
    - ejecutar en consola:
       python simulator.py --help
       python simulator.py other_config.toml
       ipython simulator.py

Scripting/Interactive use:
    $ ipython
    In [1]: import simulator
    In [2]: rodales = simulator.main(['-s'])

Funciones principales:
    - generate: generar rodales con distintos planes de manejo (necesita config & models)
    - get_models: leer modelos de crecimiento desde un archivo csv
    - read_toml: leer configuración desde un archivo toml
    - calc_biomass: calcular la biomasa para un model y una edad
    - generar_codigo_kitral: generar un diccionario de códigos Kitral basado en la Especie, edad y condición
    - write: escribir archivos de salida
    - print_manejos_possibles: imprimir los manejos posibles

Funciones auxiliares (see auxiliary.py):
    - plot_models: graficar modelos de crecimiento
    - solve_numeric: resolver numericamente los zeros de cada ecuación de crecimiento
    - solve_symbolic: resolver simbolicamente la ecuación de crecimiento, calcular los zeros
    - append_zeros: agregar ceros a los modelos

Notice: Numpy is set to print only one decimal digit
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np

np.set_printoptions(precision=1)

if "IPython" in sys.modules:
    from IPython.display import display
else:
    # Here be dragons
    display = print


def get_models(filepath="tabla.csv"):
    """Read growth models from a csv file

    Some handy introspections:
    models.dtype.names
    model.dtype.names
    dict(zip(model.dtype.names,model))

    id index from 0
    models[ models['id'] == num ] == models[num]

    models with raleo
    models[ models['next'] !=-1 ]['id']
    OJO -1 is assigned by default
    """
    models = np.genfromtxt(
        filepath,
        delimiter=",",
        names=True,  # id,next,Especie,Zona,DensidadInicial,SiteIndex,Manejo,Condicion,α,β,γ
        dtype=None,
        encoding="utf-8",
    )
    return models


def calc_biomass(model: np.void, e: int) -> float:
    """calcular la biomasa para un model y una edad
    Si la edad es menor que el zero del model, se redondea hacia arriba la edad donde es cero y
    Se pondera linealmente
    """
    e_up = np.ceil(model["stable_year"])
    if e < e_up:
        return e / e_up * (model["α"] * e_up ** model["β"] + model["γ"])
    else:
        return model["α"] * e ** model["β"] + model["γ"]


def generar_codigo_kitral(especie: str, edad: int, condicion: str) -> int:
    """Genera un diccionario de códigos Kitral basado en la Especie, edad y condición"""
    if especie == "pino":
        if edad <= 3:
            value = 19
        elif 3 < edad <= 11:
            value = 23 if condicion == "con manejo" else 20
        elif 11 < edad <= 17:
            value = 24 if condicion == "con manejo" else 21
        else:
            value = 25 if condicion == "con manejo" else 22
    elif especie == "eucalyptus":  # Eucalyptus
        if edad <= 3:
            value = 26
        elif 3 < edad <= 10:
            value = 27
        else:
            value = 28
    else:
        print("error, especie desconocida")
        return -9999
    return value


def obtener_datos_kitral(codigo_kitral: int) -> dict:
    """Obtiene la especie, rango de edad y condición a partir de un código Kitral."""
    if codigo_kitral in [19, 20, 21, 22, 23, 24, 25]:
        especie = "pino"
        if codigo_kitral == 19:
            edad = [0, 4]
            condicion = "sin manejo"
        elif codigo_kitral == 20:
            edad = [4 - 12]
            condicion = "sin manejo"
        elif codigo_kitral == 21:
            edad = [12 - 18]
            condicion = "sin manejo"
        elif codigo_kitral == 22:
            edad = [18, 41]
            condicion = "sin manejo"
        elif codigo_kitral == 23:
            edad = [6 - 12]
            condicion = "con manejo"
        elif codigo_kitral == 24:
            edad = [12 - 18]
            condicion = "con manejo"
        elif codigo_kitral == 25:
            edad = [18, 41]
            condicion = "con manejo"
    elif codigo_kitral in [26, 27, 28]:
        especie = "eucalyptus"
        if codigo_kitral == 26:
            edad = [0 - 4]
            condicion = "sin manejo"
        elif codigo_kitral == 27:
            edad = [4 - 11]
            condicion = "sin manejo"
        elif codigo_kitral == 28:
            edad = [11, 31]
            condicion = "sin manejo"
    else:
        return {"error": "Código Kitral desconocido"}

    return {"especie": especie, "edad": edad, "condicion": condicion}


def print_manejos_possibles(config):
    """Imprime todos los manejos posibles para los rodales"""
    manejos_posibles = []
    print("manejos posibles", end=": ")
    for cosecha, raleo in product(np.arange(*config["pino"]["cosechas"]), np.arange(*config["pino"]["raleos"])):
        if raleo > cosecha:
            continue
        print(f"(c{cosecha}, r{raleo})", end=", ")
        manejos_posibles.append([int(raleo), int(cosecha)])
    for cosecha in np.arange(*config["eucalyptus"]["cosechas"]):
        print(f"(c{int(cosecha)}, r{-1})", end=", ")
        manejos_posibles.append([-1, int(cosecha)])
    for cosecha in np.arange(*config["pino"]["cosechas"]):
        print(f"(c{cosecha}, r{-1})", end=", ")
        manejos_posibles.append([-1, int(cosecha)])

    for raleo in np.arange(*config["pino"]["raleos"]):
        print(f"(c{-1}, r{raleo})", end=", ")
        manejos_posibles.append([int(raleo), -1])

    print()
    return manejos_posibles


def read_toml(config_toml="config.toml"):
    if sys.version_info >= (3, 11):
        import tomllib

        with open(config_toml, "rb") as f:
            config = tomllib.load(f)
    else:
        import toml

        config = toml.load(config_toml)
    return config


def generate_random_forest(config=None, models=None):
    # 0 setup random number generator
    if seed := config["random"].get("seed"):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # 1 generate rodales
    rodales = []
    # itera = iter(range(config["rodales"]))
    # r = next(itera)
    for r in range(config["rodales"]):
        model = rng.choice(models)
        # model = rng.choice(models)
        # print(model)
        e0 = rng.integers(*config["random"]["edades"])
        e1 = e0 + config["horizonte"]
        ha = rng.integers(*config["random"]["has"])
        rodal = {
            "rid": r,
            "mid": model["id"],
            "edad_inicial": e0,
            "edad_final": e1,
            "ha": ha,
        }
        rodales += [rodal]
        display(rodal)
    return rodales


def generate_forest(config=None, filepath="bosque_data.csv"):
    data = np.genfromtxt(filepath, delimiter=",", names=True)
    rodales = []
    for r in data:
        e0 = r["edad_inicial"]
        e1 = e0 + config["horizonte"]
        ha = r["ha"]
        rodal = {
            "rid": r["rid"],  # Identificador único para cada rodal
            "mid": r["mid"],
            "edad_inicial": e0,
            "edad_final": e1,
            "ha": ha,
        }
        rodales.append(rodal)
        print(rodal)  # Reemplaza display(rodal) por print(rodal)
    return rodales


def generate(config=None, models=None, rodales=None):
    """Genera los rodales con las biomasas generadas por cada año, dependiendo de su manejo y edad de crecimiento, junto con la biomasa para vender y el codigo kitral"""
    for rodal in rodales:
        indices = np.where(models["id"] == rodal["mid"])[0]
        model = models[indices][0]
        # model = rng.choice(models)
        # print(model)
        e0 = rodal["edad_inicial"]
        e1 = rodal["edad_final"]
        edades = np.arange(e0, e1)
        ha = rodal["ha"]
        manejos = [
            {
                "rid": rodal["rid"],
                "cosecha": -1,
                "raleo": -1,
                "biomass": ha * np.array([calc_biomass(model, e) for e in edades]),
                "edades": edades,
                "eventos": ["" for e in edades],
                "vendible": [0 for e in edades],
                "codigo_kitral": [generar_codigo_kitral(model["Especie"], e, "sin manejo") for e in edades],
            }
        ]

        # has cosecha if any of the proposed "cosechas" ranges are in the simulated "edades"
        has_cosecha = any(np.isin(np.arange(*config[model["Especie"]]["cosechas"]), edades))

        # can have raleo only if it's pino
        if model["Especie"] == "pino":
            if not has_cosecha:
                # easy case
                has_raleo = (model["next"] != -1) and any(np.isin(np.arange(*config["pino"]["raleos"]), edades))
            else:
                # adjust "edades" -> "edades_manejo", by periodically "cosecha" (to harvest) via modulus operator
                # then check if raleo is in range
                has_raleo = False
                for cosecha in np.arange(*config["pino"]["cosechas"]):
                    edades_manejo = edades % cosecha
                    if model["prev"] == -1:  # no raleado desde un inicio
                        has_raleo = (model["next"] != -1) and any(
                            np.isin(np.arange(*config["pino"]["raleos"]), edades_manejo)
                        )
                    else:  # raleado desde un inicio
                        raleos = np.arange(*config["pino"]["raleos"])
                        has_raleo = any((raleo + cosecha) in edades for raleo in raleos)

                    print(cosecha, np.arange(*config["pino"]["raleos"]), edades_manejo, has_raleo)
                    if has_raleo:
                        break
        else:
            has_raleo = False

        print(f"{rodal['rid']=}, {has_cosecha=}, {has_raleo=}")
        # 4 cases combinations of "has_cosecha" and "has_raleo"
        # 1 no hacer nada
        if not has_cosecha and not has_raleo:
            # done in manejos definition
            display(manejos)
        # 2
        elif has_cosecha and not has_raleo:
            # iterb = iter(np.arange(*config[model["Especie"]]["cosechas"]))
            # cosecha = next(iterb)
            for cosecha in np.arange(*config[model["Especie"]]["cosechas"]):
                if cosecha not in edades:
                    display(f"skipping: {e0=} !< {cosecha=} !< {e1=}")
                    continue
                edades_manejo = edades % cosecha
                if model["prev"] == -1:
                    mods = [model["id"]] * len(edades_manejo)
                else:
                    mods = [
                        model["id"] if e > config[model["Especie"]]["min_ral"] else model["prev"] for e in edades_manejo
                    ]
                manejo = {
                    "rid": rodal["rid"],
                    "cosecha": cosecha,
                    "raleo": -1,
                    "biomass": ha * np.array([calc_biomass(models[m], e) for m, e in zip(mods, edades_manejo)]),
                    "edades": edades_manejo,
                    "eventos": ["c" if e == 0 else "" for e in edades_manejo],
                    "vendible": ha * np.array([calc_biomass(model, cosecha) if e == 0 else 0 for e in edades_manejo]),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], cosecha, "sin manejo")
                            if e == 0
                            else generar_codigo_kitral(model["Especie"], e, "sin manejo")
                        )
                        for e in edades_manejo
                    ],
                }
                manejos += [manejo]
                display(manejo)
        # 3
        elif not has_cosecha and has_raleo:
            # iterc = iter(np.arange(*config[model["Especie"]]["raleos"]))
            # raleo = next(iterc)
            for raleo in np.arange(*config[model["Especie"]]["raleos"]):
                if raleo not in edades:
                    display(f"skipping: {e0=} !< {raleo=} !< {e1=}")
                    continue
                manejo = {
                    "rid": rodal["rid"],
                    "cosecha": -1,
                    "raleo": raleo,
                    "biomass": ha
                    * np.array(
                        [
                            calc_biomass(model, e) if e <= raleo else calc_biomass(models[model["next"]], e)
                            for e in edades
                        ]
                    ),
                    "edades": edades,
                    "eventos": ["r" if e == raleo else "" for e in edades],
                    "vendible": ha
                    * np.array(
                        [
                            (
                                (calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo))
                                if e == raleo and models[model["next"]]["stable_year"] == 0
                                else (
                                    (
                                        (
                                            raleo
                                            / np.ceil(models[model["next"]]["stable_year"])
                                            * (
                                                model["α"] * np.ceil(models[model["next"]]["stable_year"]) ** model["β"]
                                                + model["γ"]
                                            )
                                        )
                                        - calc_biomass(models[model["next"]], raleo)
                                    )
                                    if e == raleo and models[model["next"]]["stable_year"] != 0
                                    else 0
                                )
                            )
                            for e in edades
                        ]
                    ),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "sin manejo")
                            if e < raleo
                            else generar_codigo_kitral(model["Especie"], e, "con manejo")
                        )
                        for e in edades
                    ],
                }
                manejos += [manejo]
                display(manejo)
        # 4
        else:  # has cosecha and raleo, se asume que se raleo antes del periodo 0 en calc_biomass
            # iterd = iter(
            #     product(
            #         np.arange(*config[model["Especie"]]["cosechas"]), np.arange(*config[model["Especie"]]["raleos"])
            #     )
            # )
            # cosecha, raleo = next(iterd)
            for cosecha, raleo in product(
                np.arange(*config[model["Especie"]]["cosechas"]), np.arange(*config[model["Especie"]]["raleos"])
            ):
                edades_manejo = edades % cosecha
                if model["prev"] == -1:
                    if (raleo >= cosecha) or (cosecha not in edades) or (raleo not in edades_manejo):
                        # display(f"skipping: {min(edades_manejo)=} {max(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                        continue
                    else:
                        # display(f"NO skipping: {min(edades_manejo)=} {max(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                        pass

                    mods = [model["id"] if e < raleo else model["next"] for e in edades_manejo]
                    display(f"{mods=}")
                    eventos = []
                    vendible = []
                    for e in edades_manejo:
                        if e == raleo:
                            eventos += ["r"]
                            vendible += [
                                (
                                    calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo)
                                    if models[model["next"]]["stable_year"] == 0
                                    else raleo
                                    / np.ceil(models[model["next"]]["stable_year"])
                                    * (
                                        model["α"] * np.ceil(models[model["next"]]["stable_year"]) ** model["β"]
                                        + model["γ"]
                                    )
                                    - calc_biomass(models[model["next"]], raleo)
                                )
                            ]
                        elif e == 0:
                            eventos += ["c"]
                            vendible += [calc_biomass(models[model["next"]], cosecha)]
                        else:
                            eventos += [""]
                            vendible += [0]
                else:  # si tiene prev
                    if (
                        (raleo >= cosecha)
                        or (cosecha not in edades)
                        or (raleo not in edades_manejo)
                        or (cosecha + raleo not in edades)
                    ):
                        # display(f"skipping: {min(edades_manejo)=} {max(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                        continue
                    else:
                        # display(f"NO skipping: {min(edades_manejo)=} {max(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                        pass
                    mods = [model["id"] if e >= raleo else model["prev"] for e in edades_manejo]
                    display(f"{mods=}")
                    eventos = []
                    vendible = []
                    for e in edades_manejo:
                        if e == raleo and "c" in eventos:
                            eventos += ["r"]
                            vendible += [
                                (
                                    calc_biomass(models[model["prev"]], raleo) - calc_biomass(model, raleo)
                                    if model["stable_year"] == 0
                                    else raleo
                                    / model["stable_year"]
                                    * (
                                        models[model["prev"]]["α"] * model["stable_year"] ** models[model["prev"]]["β"]
                                        + models[model["prev"]]["γ"]
                                    )
                                    - calc_biomass(model, raleo)
                                )
                            ]
                        elif e == 0:
                            eventos += ["c"]
                            vendible += [calc_biomass(model, cosecha)]
                        else:
                            eventos += [""]
                            vendible += [0]
                manejo = {
                    "rid": rodal["rid"],
                    "cosecha": cosecha,
                    "raleo": raleo,
                    "biomass": ha
                    * np.array([0 if e == 0 else calc_biomass(models[m], e) for m, e in zip(mods, edades_manejo)]),
                    "edades": edades_manejo,
                    "eventos": eventos,
                    "vendible": ha * np.array(vendible),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "con manejo")
                            if e >= raleo
                            else (
                                generar_codigo_kitral(model["Especie"], cosecha, "con manejo")
                                if e == 0
                                else generar_codigo_kitral(model["Especie"], e, "sin manejo")
                            )
                        )
                        for e in edades_manejo
                    ],
                }
                manejos += [manejo]
                display(manejo)
        rodal["manejos"] = manejos
        display(manejos)
    display(rodales)
    return rodales


def write(rodales):
    """Crea los csv de salida, con la biomasa, eventos, biomasa vendible y codigos de kitral"""
    bm = np.array([manejo["biomass"] for rodal in rodales for manejo in rodal["manejos"]])
    ev = np.array([manejo["eventos"] for rodal in rodales for manejo in rodal["manejos"]])
    vd = np.array([manejo["vendible"] for rodal in rodales for manejo in rodal["manejos"]])
    kt = np.array([manejo["codigo_kitral"] for rodal in rodales for manejo in rodal["manejos"]])
    names = ",".join(
        [f"R{rodal['rid']}_c{manejo['cosecha']}_r{manejo['raleo']}" for rodal in rodales for manejo in rodal["manejos"]]
    )
    names = names.replace("_r-1", "").replace("_c-1", "")
    np.savetxt("biomass.csv", bm.T, delimiter=",", header=names, comments="")
    np.savetxt("events.csv", ev.T, delimiter=",", header=names, comments="", fmt="%s")
    np.savetxt("vendible.csv", vd.T, delimiter=",", header=names, comments="")
    np.savetxt("codigo_kitral.csv", kt.T, delimiter=",", header=names, comments="", fmt="%s")

    bos_names = ["rid", "mid", "edad_inicial", "ha"]  # aprender hacer formato decente
    bos = np.array([tuple(r[k] for k in bos_names) for r in rodales])
    np.savetxt("bosque.csv", bos, delimiter=",", header=",".join(bos_names), comments="", fmt="%d")


def arg_parser(argv=None):
    """Parse command line arguments."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="More at https://fire2a.github.io/fire2a-lib",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        type=Path,
        help="Configuration of simulation parameters file",
        default="config.toml",
    )
    parser.add_argument(
        "-m",
        "--models_table",
        type=Path,
        help="Table of growth models.csv",
        default="tabla.csv",
    )
    parser.add_argument(
        "-nw",
        "--no_write",
        action="store_true",
        help="Do not write output files (see write method)",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--data_forest",
        type=Path,
        help="Data of the forest",
        default="./bosque_data.csv",
    )
    parser.add_argument(
        "-s",
        "--script",
        action="store_true",
        help="Run in script mode, returning the rodales object. Example: import simulator; rodales = simulator.main(['-s','-nw'])",
        default=False,
    )
    parser.add_argument("-r", "--random", action="store_true", help="Create the forest with random data", default=False)

    args = parser.parse_args(argv)
    if Path(args.config_file).is_file() is False:
        parser.error(f"File {args.config_file} not found")
    if Path(args.models_table).is_file() is False:
        parser.error(f"File {args.models_table} not found")
    return args


def main(argv=None):
    """Main entry point for command line usage.

    args = arg_parser(["config.toml", "-m", "tabla.csv"])
    args = arg_parser(None)
    """
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)
    print("Parsed arguments", args)

    # 1 read config.toml
    config = read_toml(args.config_file)

    # 2 read models
    models = get_models(args.models_table)

    # 3 generate rodales
    if args.random:
        rodales_sin_manejo = generate_random_forest()

    else:
        # usar bosque_data.csv, si no se tiene se puede crear con las funciones del auxiliary
        rodales_sin_manejo = generate_forest(config, args.data_forest)

    rodales = generate(config, models, rodales_sin_manejo)

    # 4 write output files
    if not args.no_write:
        write(rodales)

    # 5 return rodales if scripting
    if args.script:
        return rodales

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
