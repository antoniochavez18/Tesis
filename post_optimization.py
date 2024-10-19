import ast
import os
import re
import pandas as pd
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


def filtro(rodales):
    # Cargamos los datos de los archivos
    df_soluciones = pd.read_csv("soluciones_x.csv")  # Archivo de soluciones

    # Parámetros
    RR = config["rodales"]  # Cantidad de rodales

    # Lista para almacenar las soluciones
    soluciones = []

    # Iterar sobre cada solución (por columnas, ignorando la primera columna de nombres)
    for col_idx, col_name in enumerate(df_soluciones.columns[1:]):
        solucion = {}  # Inicializamos un diccionario por solución

        # Iterar sobre cada rodal
        for r in range(RR):
            # Obtener la política
            pol = ast.literal_eval(df_soluciones[col_name].iloc[r])

            # Inicializamos un diccionario por rodal para almacenar "codigo_kitral" y "vendible"
            rodal_data = {}

            if pol == 0:
                # Si no hay raleo ni cosecha, usar el primer manejo
                rodal_data["codigo_kitral"] = rodales[r]["manejos"][0]["codigo_kitral"]
                rodal_data["vendible"] = rodales[r]["manejos"][0]["vendible"]
            else:
                # Buscar el manejo que coincida con la política
                for m in range(len(rodales[r]["manejos"])):
                    if rodales[r]["manejos"][m]["raleo"] == pol[0] and rodales[r]["manejos"][m]["cosecha"] == pol[1]:
                        rodal_data["codigo_kitral"] = rodales[r]["manejos"][m]["codigo_kitral"]
                        rodal_data["vendible"] = rodales[r]["manejos"][m]["vendible"]

            # Guardamos el diccionario de datos del rodal en la solución correspondiente
            solucion[r] = rodal_data

        # Añadir la solución a la lista de soluciones
        soluciones.append(solucion)

    return soluciones


def id2xy(idx: int, w: int, h: int) -> tuple[int, int]:
    """Transform a pixel or cell index, into x,y coordinates.
    In GIS, the origin is at the top-left corner, read from left to right, top to bottom.  
    If your're used to matplotlib, the y-axis is inverted.  
    Also as numpy array, the index of the pixel is [y, x].

    Args:
        param idx: index of the pixel or cell (0,..,w*h-1)  
        param w: width of the image or grid  
        param h: height of the image or grid (not really used!)

    Returns:
        tuple: (x, y) coordinates of the pixel or cell  
    """  # fmt: skip
    return idx % w, idx // w


def xy2id(x: int, y: int, w: int) -> int:
    """Transform a x,y coordinates into a pixel or cell index.
    In GIS, the origin is at the top-left corner, read from left to right, top to bottom.  
    If your're used to matplotlib, the y-axis is inverted.  
    Also as numpy array, the index of the pixel is [y, x].

    Args:
        param x: width or horizontal coordinate of the pixel or cell  
        param y: height or vertical coordinate of the pixel or cell  
        param w: width of the image or grid  

    Returns:
        int: index of the pixel or cell (0,..,w\*h-1)
    """  # fmt: skip
    return y * w + x


def lista_a_matriz(lista, w, h):
    # Crear una matriz vacía de tamaño h (filas) x w (columnas) con tipo 'object' para soportar strings
    matriz = np.empty((h, w), dtype=int)  # Usamos dtype=object para strings y otros tipos de datos

    # Iterar sobre la lista y colocar cada elemento en su posición en la matriz
    for idx, valor in enumerate(lista):
        x, y = id2xy(idx, w, h)  # Obtener las coordenadas (x, y) para el índice
        matriz[y][x] = valor  # Asignar el valor en la posición correspondiente en la matriz

    return matriz


def matriz_cod(soluciones, w, h):
    RR = config["rodales"]  # Cantidad de rodales
    periodos = config["horizonte"]
    matrices = []  # Aquí almacenaremos las matrices organizadas por solución y periodo

    for s in range(len(soluciones)):  # Para cada solución
        matrices_por_solucion = []  # Lista para matrices de cada solución

        for t in range(periodos):  # Para cada periodo
            lista_kitral = []  # Lista para almacenar los códigos kitral de todos los rodales en un periodo

            for r in range(RR):  # Para cada rodal
                # Obtener el código kitral para el rodal 'r' en el periodo 't'
                lista_kitral.append(soluciones[s][r]["codigo_kitral"][t])

            # Convertir la lista de códigos kitral en una matriz usando la función lista_a_matriz
            matriz_kitral = lista_a_matriz(lista_kitral, w, h)

            # Almacenar la matriz en la lista correspondiente a la solución actual
            matrices_por_solucion.append(matriz_kitral)

        # Almacenar la lista de matrices de la solución 's' en la lista principal
        matrices.append(matrices_por_solucion)

    return matrices  # Devuelve una lista de matrices por solución y periodo


def matriz_a_asc(matriz, nombre_archivo, xllcorner=457900, yllcorner=5716800, cellsize=100, nodata_value=-9999):
    """
    Función para guardar una matriz como un archivo .asc (ASCII Grid).

    Args:
    - matriz: numpy array 2D que contiene los valores del grid.
    - nombre_archivo: el nombre del archivo de salida (incluyendo .asc).
    - xllcorner: coordenada X de la esquina inferior izquierda (por defecto 0).
    - yllcorner: coordenada Y de la esquina inferior izquierda (por defecto 0).
    - cellsize: tamaño de cada celda en el grid (por defecto 1).
    - nodata_value: valor para las celdas que no contienen datos (por defecto -9999).
    """
    nrows, ncols = matriz.shape  # Obtener las dimensiones de la matriz

    with open(nombre_archivo, "w") as f:
        # Escribir el encabezado del archivo .asc
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner}\n")
        f.write(f"yllcorner     {yllcorner}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata_value}\n")

        # Escribir los valores de la matriz
        for row in matriz:
            f.write(" ".join(map(str, row)) + "\n")


def leer_matriz_asc(file_path):
    """Lee un archivo .asc y devuelve su matriz de datos como un array de numpy."""
    with open(file_path, "r") as file:
        asc_data = file.readlines()

    # Extraemos las líneas de datos, asumiendo que el encabezado tiene 6 líneas
    data_start_index = 6
    matrix_data = asc_data[data_start_index:]

    # Convertimos las líneas en una matriz
    matrix = [list(map(float, line.split())) for line in matrix_data]

    return np.array(matrix)


def leer_matrices_de_carpeta(carpeta_path, num_soluciones, num_periodos):
    """Lee todos los archivos .asc en la carpeta y organiza las matrices en una lista bidimensional.

    Args:
        carpeta_path (str): La ruta a la carpeta que contiene los archivos .asc.
        num_soluciones (int): Número total de soluciones.
        num_periodos (int): Número total de periodos.

    Returns:
        list: Lista bidimensional con matrices organizadas como m[solucion][periodo].
    """
    # Crear una lista bidimensional para almacenar las matrices
    matrices = [[None for _ in range(num_periodos)] for _ in range(num_soluciones)]

    # Expresión regular para extraer solucion y periodo de los nombres de los archivos
    pattern = r"solucion_(\d+)_periodo_(\d+)\.asc"

    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(carpeta_path):
        if filename.endswith(".asc"):
            match = re.search(pattern, filename)
            if match:
                solucion = int(match.group(1))
                periodo = int(match.group(2))

                # Leer la matriz del archivo
                file_path = os.path.join(carpeta_path, filename)
                matriz = leer_matriz_asc(file_path)

                # Almacenar la matriz en la lista correspondiente
                matrices[solucion][periodo] = matriz

    return matrices


def multiplicar_listas(bp, filtro):
    soluciones = len(bp)  # Número de soluciones (dimensión s)
    rodales = len(bp[0])  # Número de rodales (dimensión r)
    periodos = len(bp[0][0])  # Número de periodos (dimensión t)

    # Crear una lista vacía para almacenar los resultados de la multiplicación
    resultado = [[[0 for t in range(periodos)] for r in range(rodales)] for s in range(soluciones)]

    # Iterar sobre soluciones, rodales y periodos
    for s in range(soluciones):
        for r in range(rodales):
            for t in range(periodos):
                # Multiplicar el valor correspondiente de cada lista
                resultado[s][r][t] = bp[s][r][t] * filtro[s][r]["vendible"][t]

    return resultado


def sumar_por_solucion(ganancias_totales):
    soluciones = len(ganancias_totales)  # Número de soluciones
    rodales = len(ganancias_totales[0])  # Número de rodales
    periodos = len(ganancias_totales[0][0])  # Número de periodos

    # Crear una lista para almacenar los totales por solución
    total = [0 for _ in range(soluciones)]

    # Iterar sobre soluciones, rodales y periodos para sumar
    for s in range(soluciones):
        suma_solucion = 0  # Inicializamos la suma para cada solución
        for r in range(rodales):
            for t in range(periodos):
                # Sumamos todos los valores en ganancias_totales para la solución s
                suma_solucion += ganancias_totales[s][r][t]

        total[s] = suma_solucion  # Guardamos la suma total para la solución s

    return total
