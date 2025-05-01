import csv
import sys

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

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


def generate_random_walk_prices(initial_price, num_periods, mu=0.05, sigma=0.1):
    """Genera precios futuros usando un random walk con drift."""
    prices = [initial_price]
    rng = np.random.default_rng(config["random"]["seed"])
    for _ in range(1, num_periods):
        drift = mu - 0.5 * sigma**2
        shock = sigma * rng.normal()
        next_price = prices[-1] * np.exp(drift + shock)
        prices.append(next_price)
    plt.figure(figsize=(10, 6))
    plt.plot(prices, linestyle="-", label="precios")
    plt.title("Precios por periodo")
    plt.xlabel("Períodos")
    plt.ylabel("Precio")
    plt.legend(title="Precios")
    plt.grid(True)
    # plt.savefig("precios_por_periodo.png")
    # plt.show()
    return prices


def calc_biomass_0(rodales):
    biom_0 = 0
    for r in range(len(rodales)):
        biom_0 += rodales[r]["manejos"][0]["biomass"][0]
    return biom_0


def no_poli(rodales):
    no_pol = []
    for r in range(len(rodales)):
        no_pol.append(rodales[r]["manejos"][0]["biomass"][-1])
    return no_pol


def model_t(rodales, politicas, prices, dataset_name, config, config_opti):
    """Modelo de optimización para maximizar el valor presente neto (NPV) de la venta de biomasa."""
    # Configuraciones y parámetros iniciales
    tasa = config_opti["opti"]["tasa"]
    biom_0 = calc_biomass_0(rodales)
    no_pol = no_poli(rodales)

    RR = len(rodales)
    periodos = config["horizonte"]

    a = [[[0 for _ in range(periodos)] for _ in range(len(politicas))] for _ in range(RR)]
    b = [[0 for _ in range(len(politicas))] for _ in range(RR)]
    rodales_sin_combinaciones = set(range(RR))
    valid_combinations = set()

    # Asignar combinaciones válidas iniciales
    for r in range(RR):
        for m in range(len(rodales[r]["manejos"])):
            r_value = rodales[r]["manejos"][m]["raleo"]
            c_value = rodales[r]["manejos"][m]["cosecha"]
            if [r_value, c_value] in politicas:
                politica = politicas.index([r_value, c_value])
                for t in range(periodos):
                    a[r][politica][t] = rodales[r]["manejos"][m]["vendible"][t]
                    b[r][politica] = rodales[r]["manejos"][m]["biomass"][-1]
                    if any(a[r][politica][t] != 0 for t in range(periodos)):
                        valid_combinations.add((r, politica))
                        rodales_sin_combinaciones.discard(r)

    # Agregar combinaciones base para rodales sin combinaciones
    for rodal in rodales_sin_combinaciones:
        valid_combinations.add((rodal, 0))

    # Parámetros del modelo
    B = config_opti["opti"]["B"]
    C = config_opti["opti"]["C"]
    D = config_opti["opti"]["D"]

    R = list(range(RR))
    M = list(range(len(politicas)))
    H = list(range(periodos))

    # Crear el modelo de optimización
    model = gp.Model()
    model.setParam("Heuristics", 0.1)
    model.setParam("MIPGap", 0.01)
    model.setParam("VarBranch", 1)
    model.setParam("Cuts", 1)
    model.setParam("Presolve", 1)
    # model.setParam("PumpPasses", 10)

    # Variables
    x = model.addVars(valid_combinations, vtype=GRB.BINARY)
    v = model.addVars(H, vtype=GRB.CONTINUOUS)
    y = model.addVars(R, vtype=GRB.CONTINUOUS)

    print(f"Cantidad de variables binarias x: {len(valid_combinations)}")
    print(f"Cantidad de variables continuas v: {len(H)}")
    print(f"Cantidad de variables continuas y: {len(R)}")

    # Función objetivo
    npv = gp.quicksum(x[i, j] * prices[t] * a[i][j][t] / (1 + tasa) ** t for (i, j) in valid_combinations for t in H)
    model.setObjective(npv, GRB.MAXIMIZE)

    # Restricciones
    # 4.5
    model.addConstr(gp.quicksum(x[i, j] * C for (i, j) in valid_combinations) <= B)
    # 4.1
    for i in R:
        model.addConstr(gp.quicksum(x[i, j] for (i_, j) in valid_combinations if i_ == i) <= 1)
    # 4.3
    for t in H:
        model.addConstr(gp.quicksum(x[i, j] * a[i][j][t] for (i, j) in valid_combinations) == v[t])
        if t >= 1:
            # 4.4
            model.addConstr(-v[t - 1] + v[t] <= v[t - 1] / 10)
            model.addConstr(-v[t - 1] + v[t] >= -v[t - 1] / 10)
            # 4.2
            model.addConstr(v[t] >= D)

    for i in R:
        y_expr = gp.quicksum(b[i][j] * x[i, j] for (i_, j) in valid_combinations if i_ == i)
        policy_selected = gp.quicksum(x[i, j] for (i_, j) in valid_combinations if i_ == i)
        # 4.6
        model.addConstr(y[i] == y_expr + (1 - policy_selected) * no_pol[i])
    # 4.7
    model.addConstr(gp.quicksum(y[i] for i in R) >= biom_0)

    # Registro del progreso del objetivo y gap
    all_obj_vals = []
    all_gaps = []

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Captura el valor del objetivo en soluciones factibles
            obj_vals.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        if where == GRB.Callback.MIP:
            # Captura el GAP actual en cada iteración del MIP
            obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
            if obj_best > 0:  # Evita divisiones por cero
                gap = abs((obj_bound - obj_best) / obj_best) * 100
                gaps.append(gap)

    # Optimización del modelo
    obj_vals = []
    gaps = []
    model.optimize(callback)

    # Almacenar los valores del objetivo y GAP para la solución base
    all_obj_vals.append(obj_vals.copy())
    all_gaps.append(gaps.copy())

    # Optimización de la solución base
    model.optimize()

    # Almacenar la solución base
    soluciones = []  # Lista para guardar las soluciones
    soluciones_v = []  # Para almacenar los valores de v_t
    valores_objetivo = []  # Lista para guardar los valores objetivo de cada solución

    solucion_generada = {key: x[key].X for key in valid_combinations if x[key].X > 0.9}
    soluciones.append(solucion_generada.copy())  # Guardar copia de la solución base
    soluciones_v.append([v[t].X for t in H])  # Guardar los valores de v_t para la solución base
    valores_objetivo.append(model.ObjVal)  # Guardar el valor objetivo de la solución base

    # Generar soluciones adicionales con restricciones de diversidad
    num_cambios = int(len(rodales) * 0.1)  # Cambiar un 10% de los rodales

    for sol_num in range(1, config_opti["opti"]["soluciones"]):  # Generar soluciones adicionales
        obj_vals = []
        gaps = []

        # Agregar restricciones de diversidad respecto a soluciones previas
        for sol_prev in soluciones:
            combinaciones_previas = list(sol_prev.keys())
            model.addConstr(
                gp.quicksum(x[i, j] for (i, j) in combinaciones_previas) <= len(combinaciones_previas) - num_cambios
            )

        # Optimizar el modelo con las restricciones de diversidad
        model.update()
        model.optimize(callback)

        # Guardar la nueva solución generada
        solucion_generada = {key: x[key].X for key in valid_combinations if x[key].X > 0.9}
        soluciones.append(solucion_generada.copy())  # Guardar copia de la solución generada
        soluciones_v.append([v[t].X for t in H])  # Guardar los valores de v_t
        valores_objetivo.append(model.ObjVal)  # Guardar el valor objetivo de la solución generada

        # Almacenar los valores del objetivo y GAP para la solución actual
        all_obj_vals.append(obj_vals.copy())
        all_gaps.append(gaps.copy())

    # Generar gráfico del progreso del valor objetivo para todas las soluciones
    plt.figure(figsize=(10, 6))
    for sol_num, obj_vals in enumerate(all_obj_vals):
        plt.plot(range(len(obj_vals)), obj_vals, linestyle="-", label=f"Solución {sol_num + 1}")
    plt.title(f"Progreso del valor objetivo durante la optimización ({dataset_name})")
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor objetivo")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"progreso_valor_objetivo_{dataset_name}.png")
    # plt.show()

    # Generar gráfico de la evolución del GAP para todas las soluciones
    plt.figure(figsize=(10, 6))
    for sol_num, gaps in enumerate(all_gaps):
        plt.plot(range(len(gaps)), gaps, linestyle="-", label=f"Solución {sol_num + 1}")
    plt.title(f"Evolución del GAP durante la optimización ({dataset_name})")
    plt.xlabel("Iteraciones")
    plt.ylabel("GAP (%)")
    plt.yscale("log")  # Escala logarítmica para mejor visualización
    plt.legend()
    plt.grid(True)
    plt.savefig(f"evolucion_gap_{dataset_name}.png")
    # plt.show()

    # Generar gráfico de v_t * price para las soluciones en valor presente
    plt.figure(figsize=(10, 6))
    for sol_num, valores_v in enumerate(soluciones_v):
        valores_v_presente = [val * prices[t] / (1 + tasa) ** t for t, val in enumerate(valores_v)]
        plt.plot(H, valores_v_presente, marker="o", linestyle="-", label=f"Solución {sol_num + 1}")
    plt.title(f"Valores presentes por período para cada solución en valor presente ({dataset_name})")
    plt.xlabel("Períodos")
    plt.ylabel("v_t * Precio (Ventas en valor presente)")
    plt.legend(title="Soluciones")
    plt.grid(True)
    plt.savefig(f"valores_presentes_por_solucion_{dataset_name}.png")
    # plt.show()

    # Imprimir todas las soluciones generadas
    for sol_idx, sol in enumerate(soluciones):
        print(f"\nSolución {sol_idx + 1}:")
        for rodal, manejo in sol.keys():
            print(f"Rodal {rodales[rodal]['rid']}, Manejo {politicas[manejo]}")

    # Después de optimizar el modelo, guarda las soluciones en una lista
    solutions = []
    for sol in soluciones:
        solution_dict = {}
        for i, j in sol.keys():
            solution_dict[i] = politicas[j]
        solutions.append(solution_dict)

    # Crear una lista de filas para el CSV, donde cada fila es un rodal y las columnas son las soluciones
    csv_rows = []
    headers = ["ID Rodal"] + [f"Solucion_{s + 1}" for s in range(len(solutions))]

    # Iterar sobre cada rodal usando su ID
    for i in range(RR):  # Itera sobre cada índice de rodal
        rodal_id = rodales[i]["rid"]  # Obtén el ID del rodal
        row = [rodal_id]  # Inicia la fila con el ID del rodal
        for sol in solutions:
            policy = sol.get(i, 0)  # Obtén la política seleccionada para el rodal i (si no hay, devuelve 0)
            row.append(policy)  # Agrega la política seleccionada o 0 si no hay
        csv_rows.append(row)

    # Guardar en un archivo CSV
    csv_filename = f"soluciones_{dataset_name}.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Escribir los encabezados
        writer.writerows(csv_rows)  # Escribir las filas con los datos

    print(f"Las soluciones de x[i,j] se han guardado en el archivo {csv_filename} con los IDs de los rodales.")

    return valores_objetivo, csv_rows


def model_t_cplex(rodales, politicas, prices, dataset_name, config, config_opti):
    from docplex.mp.model import Model

    tasa = config_opti["opti"]["tasa"]
    biom_0 = calc_biomass_0(rodales)
    no_pol = no_poli(rodales)

    RR = len(rodales)
    periodos = config["horizonte"]

    a = [[[0 for _ in range(periodos)] for _ in range(len(politicas))] for _ in range(RR)]
    b = [[0 for _ in range(len(politicas))] for _ in range(RR)]
    rodales_sin_combinaciones = set(range(RR))
    valid_combinations = set()

    for r in range(RR):
        for m, manejo in enumerate(rodales[r]["manejos"]):
            r_value = manejo["raleo"]
            c_value = manejo["cosecha"]
            if [r_value, c_value] in politicas:
                politica = politicas.index([r_value, c_value])
                for t in range(periodos):
                    a[r][politica][t] = manejo["vendible"][t]
                    b[r][politica] = manejo["biomass"][-1]
                    if any(a[r][politica][t] != 0 for t in range(periodos)):
                        valid_combinations.add((r, politica))
                        rodales_sin_combinaciones.discard(r)

    for rodal in rodales_sin_combinaciones:
        valid_combinations.add((rodal, 0))

    B = config_opti["opti"]["B"]
    C = config_opti["opti"]["C"]
    D = config_opti["opti"]["D"]

    R = list(range(RR))
    H = list(range(periodos))

    soluciones = []
    soluciones_v = []
    valores_objetivo = []

    mdl = Model(name="forest_optimization")
    mdl.parameters.mip.tolerances.mipgap = 0.01
    mdl.parameters.mip.strategy.heuristicfreq = 1
    mdl.parameters.mip.cuts.covers = -1
    mdl.parameters.preprocessing.presolve = 1

    x = mdl.binary_var_dict(valid_combinations, name="x")
    v = mdl.continuous_var_list(H, name="v")
    y = mdl.continuous_var_list(R, name="y")

    npv = mdl.sum(x[i, j] * prices[t] * a[i][j][t] / (1 + tasa) ** t for (i, j) in valid_combinations for t in H)
    mdl.maximize(npv)

    mdl.add_constraint(mdl.sum(x[i, j] * C for (i, j) in valid_combinations) <= B)

    for i in R:
        mdl.add_constraint(mdl.sum(x[i, j] for (i_, j) in valid_combinations if i_ == i) <= 1)

    for t in H:
        mdl.add_constraint(mdl.sum(x[i, j] * a[i][j][t] for (i, j) in valid_combinations) == v[t])
        if t >= 1:
            mdl.add_constraint(v[t] - v[t - 1] <= v[t - 1] / 10)
            mdl.add_constraint(v[t - 1] - v[t] <= v[t - 1] / 10)
        mdl.add_constraint(v[t] >= D)

    for i in R:
        y_expr = mdl.sum(b[i][j] * x[i, j] for (i_, j) in valid_combinations if i_ == i)
        policy_selected = mdl.sum(x[i, j] for (i_, j) in valid_combinations if i_ == i)
        mdl.add_constraint(y[i] == y_expr + (1 - policy_selected) * no_pol[i])

    mdl.add_constraint(mdl.sum(y[i] for i in R) >= biom_0)

    num_solutions = config_opti["opti"]["soluciones"]
    num_cambios = int(len(rodales) * 0.1)

    for sol_num in range(num_solutions):
        if sol_num > 0:
            for sol_prev in soluciones:
                combinaciones_previas = list(sol_prev.keys())
                mdl.add_constraint(
                    mdl.sum(x[i, j] for (i, j) in combinaciones_previas) <= len(combinaciones_previas) - num_cambios
                )

        sol = mdl.solve(log_output=False)

        if not sol:
            print(f"No se encontró solución en la iteración {sol_num + 1}.")
            break

        solucion_generada = {
            (i, j): x[i, j].solution_value for (i, j) in valid_combinations if x[i, j].solution_value > 0.9
        }
        soluciones.append(solucion_generada.copy())
        soluciones_v.append([v[t].solution_value for t in H])
        valores_objetivo.append(sol.objective_value)

    # Guardar resultados CSV con detalles del manejo utilizado
    csv_rows = []
    headers = ["ID Rodal"] + [f"Solucion_{s + 1}" for s in range(len(soluciones))]
    for i in range(RR):
        rodal_id = rodales[i]["rid"]
        row = [rodal_id]
        for sol in soluciones:
            selected_policy = None
            for j in range(len(politicas)):
                if (i, j) in sol:
                    selected_policy = politicas[j]
                    break
            row.append(selected_policy if selected_policy else 0)
        csv_rows.append(row)

    csv_filename = f"soluciones_{dataset_name}.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    return valores_objetivo, csv_rows
