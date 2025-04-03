# Tesis
# Overview
Se realiza metodologia para planificar los manejos de un predio forestal bajo incendios y construyendo cortafuegos, para maximizar ganancia en un horizonte de 10 anos.

# Trama
# 1. Preparacion ambiente

1.1. Clonar
git@github.com:antoniochavez18/Tesis.git tesis

1.2. Clonar el growth simulator.py, auxiliary.py y tabla.csv de git@github.com:fire2a/growth.git

    # git clone git@github.com:fire2a/growth.git
    # cp growth/growth_simulator.py .
    # cp growth/auxiliary.py .
    # cp growth/tabla.csv .

1.3. Instalar dependencias

    # instale qgis: https://fire2a.github.io/docs/qgis-management/install.html
    # or
    # i known what i'm doing:
    python3 -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install -r requirements.txt

# 2. Ejecutar 

    - Primero se simula el crecimiento del bosque, 
    - luego se crean distintas opciones de cortafuegos  
        - quemando, calculando el dpv x periodo y calculando el valor presente de todo; luego para elegir cortafuegos optimizar knapsack con valor dpv restringiendo 1,2 y 3% de area tratada (cortafuego)
    - se elige el mejor: analisis de sensibilidad (x variacion de area tratada) de cortafuegos usando NPE: net protection effect, eliges el de mayor NPE
    - se optimizan los manejos,
        - utilizando 5 soluciones con y sin cortafuego (10)
    - se queman (simula incendios) las 10 soluciones cada anyo del horizonte
    - se adapta el burn probability para acumular informacion de anyos pasados
    - se actualizan las ganancias
    - se elije 1 de las 10 soluciones
    profit.

## 2.1 Simular crecimiento del bosque

    python simulator.py config.toml

