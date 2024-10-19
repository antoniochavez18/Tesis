import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import sys


if sys.version_info >= (3, 11):
    import tomllib

    with open("config_test.toml", "rb") as f:
        config = tomllib.load(f)
else:
    import toml

    config = toml.load("config.toml")
rng = np.random.default_rng(4)


def generate_random_walk_prices(initial_price, num_periods, mu=0.05, sigma=0.1):
    """Genera precios futuros usando un random walk con drift."""
    prices = [initial_price]
    for _ in range(1, num_periods):
        drift = mu - 0.5 * sigma**2
        shock = sigma * rng.normal()
        next_price = prices[-1] * np.exp(drift + shock)
        prices.append(next_price)
    return prices


def calc_biomass_0(rodales):
    biom_0 = 0
    for r in range(config["rodales"]):
        biom_0 += rodales[r]["manejos"][0]["biomass"][0]
    return biom_0


def no_poli(rodales):
    no_pol = []
    periodos = config["horizonte"]
    for r in range(config["rodales"]):
        no_pol.append(rodales[r]["manejos"][0]["biomass"][periodos - 1])
    return no_pol


def model_t(rodales, politicas, price):
    tasa = config["opti"]["tasa"]
    biom_0 = calc_biomass_0(rodales)
    no_pol = no_poli(rodales)

    RR = config["rodales"]
    periodos = config["horizonte"]
    prices = generate_random_walk_prices(price, periodos, mu=0.05, sigma=0.1)

    a = [[[0 for _ in range(periodos)] for _ in range(len(politicas))] for _ in range(RR)]
    b = [[0 for _ in range(len(politicas))] for _ in range(RR)]
    rodales_sin_combinaciones = set(range(RR))
    valid_combinations = set()
    for r in range(RR):
        for m in range(len(rodales[r]["manejos"])):
            r_value = rodales[r]["manejos"][m]["raleo"]
            c_value = rodales[r]["manejos"][m]["cosecha"]
            if [r_value, c_value] in politicas:
                politica = politicas.index([r_value, c_value])
                for t in range(periodos):
                    a[r][politica][t] = rodales[r]["manejos"][m]["vendible"][t]
                    b[r][politica] = rodales[r]["manejos"][m]["biomass"][periodos - 1]
                    if any(a[r][politica][t] != 0 for t in range(periodos)):
                        valid_combinations.add((r, politica))
                        rodales_sin_combinaciones.discard(r)

    for rodal in rodales_sin_combinaciones:
        valid_combinations.add((rodal, 0))

    B = config["opti"]["B"]
    C = config["opti"]["C"]
    D = config["opti"]["D"]

    R = list(range(RR))
    M = list(range(len(politicas)))
    H = list(range(periodos))

    model = gp.Model()

    model.setParam("Heuristics", 0.1)  # Aumenta el esfuerzo en heurísticas
    model.setParam(GRB.Param.PoolSolutions, 5)
    model.setParam(GRB.Param.PoolSearchMode, 1)
    model.setParam(GRB.Param.PoolGap, 0.1)
    model.setParam("MIPGap", 0.01)
    model.setParam("VarBranch", 1)
    model.setParam("Cuts", 1)
    model.setParam("Presolve", 1)
    # model.setParam("RINS", 5)  # Activa la heurística RINS
    # model.setParam("PumpPasses", 5)  # Ajusta la frecuencia de la heurística de Feasibility Pump

    x = model.addVars(valid_combinations, vtype=GRB.BINARY)
    v = model.addVars(H, vtype=GRB.CONTINUOUS)
    y = model.addVars(R, vtype=GRB.CONTINUOUS)

    npv = gp.quicksum(x[i, j] * prices[t] * a[i][j][t] / (1 + tasa) ** t for (i, j) in valid_combinations for t in H)
    model.setObjective(npv, GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(x[i, j] * C for (i, j) in valid_combinations) <= B)

    for i in R:
        model.addConstr(gp.quicksum(x[i, j] for (i_, j) in valid_combinations if i_ == i) <= 1)

    for t in H:
        model.addConstr(gp.quicksum(x[i, j] * a[i][j][t] for (i, j) in valid_combinations) == v[t])
        if t >= 1:
            model.addConstr(-v[t - 1] + v[t] <= v[t - 1] / 10)
            model.addConstr(-v[t - 1] + v[t] >= -v[t - 1] / 10)
            model.addConstr(v[t] >= D)

    for i in R:
        y_expr = gp.quicksum(b[i][j] * x[i, j] for (i_, j) in valid_combinations if i_ == i)

        # Añadir una nueva variable auxiliar que represente si se selecciona una política
        policy_selected = gp.quicksum(x[i, j] for (i_, j) in valid_combinations if i_ == i)

        # Si se selecciona una política, y[i] es la suma ponderada, de lo contrario es no_pol[i]
        model.addConstr(y[i] == y_expr + (1 - policy_selected) * no_pol[i])
    model.addConstr(gp.quicksum(y[i] for i in R) >= biom_0)

    # model running
    model.optimize()

    for t in H:
        print(f"v[{t}] = {v[t].X}")  # .X para acceder al valor de la variable optimizada
    y_total = sum(y[i].X for i in R)

    # Imprimir el valor total de y
    print(f"El valor total de y es: {y_total}")
    print(biom_0)
    for i in R:
        pol_ut = False
        for j in M:
            if (i, j) in x and x[i, j].X > 0.9:
                pol_ut = True
                print("Se utilizó la política", politicas[j], "para el rodal", i)
        if not pol_ut:
            print(f"El rodal {i} no utilizó ninguna política.")

    for sol in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, sol)
        print(f"\nSolución {sol + 1}:")
        for t in H:
            print(f"v[{t}] = {v[t].Xn}")  # .Xn para acceder al valor de la n-ésima solución
        print(f"Valor de la función objetivo: {model.PoolObjVal}")

    v_values = [v[t].X for t in H]

    # Graficar los valores de v[t]
    plt.figure(figsize=(10, 6))
    plt.plot(H, v_values, marker="o", linestyle="-", color="b")
    plt.title("Valor de v[t] a lo largo de los períodos")
    plt.xlabel("Períodos")
    plt.ylabel("Valor de v[t]")
    plt.grid(True)
    plt.show()

    # grafico de Van
    # Rodales
    # warm start funcion objetivo rodales
    # itertools

    # Después de optimizar el modelo, guarda las soluciones en una lista
    solutions = []
    for sol in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, sol)
        solution_dict = {}
        for i, j in valid_combinations:
            if x[i, j].Xn > 0.9:  # Si la política se selecciona
                solution_dict[i] = politicas[j]  # Guardar el índice de la política j para el rodal i
        solutions.append(solution_dict)

    # Crear una lista de filas para el CSV, donde cada fila es un rodal y las columnas son las soluciones
    csv_rows = []
    headers = ["Rodal"] + [f"Solucion_ {s + 1}" for s in range(len(solutions))]

    for i in range(RR):  # Itera sobre cada rodal
        row = [i]  # Inicia la fila con el número de rodal
        for sol in solutions:
            policy = sol.get(i, 0)  # Obtén la política seleccionada para el rodal i (si no hay, devuelve 0)
            row.append(policy)  # Agrega la política seleccionada o 0 si no hay
        csv_rows.append(row)

    # Guardar en un archivo CSV
    import csv  # Asegúrate de importar csv si no está ya

    csv_filename = "soluciones_x.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Escribir los encabezados
        writer.writerows(csv_rows)  # Escribir las filas con los datos

    print(f"Las soluciones de x[i,j] se han guardado en el archivo {csv_filename}.")

    return model.ObjVal


