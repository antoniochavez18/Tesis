# Modelo 1 forestal

### 1. Conjuntos

- **$M:$** Manejos posibles

### 2. Parametros

- H : Horizonte de simulación
- R : Cantidad de rodales en la simulación
- $A_{ijt}:$ Cantidad de biomasa retirada al rodal i con manejo j en periodo t

#### Parametros Mercado

- b2m : funcion de biomasa_a_madera(id,Especie,bio): funcion convierte segun id del rodal de biomasa a madera
[ton_madera/ton_biomasa]
- $P_{mp}$: precio madera pino [$/ton]
- $P_{me}$: precio madera euc [$/ton]
- $C_{ij}:$ Costo implementar politica j en rodal i [$/ha]
- D : Demanda anual (posible D_p y D_e) (preguntar si es por madera, biomasa o plata) [ton_madera]
- B : Presupuesto Total (todo el horizonte)

### 3. Variables

- $X_{ij}$: Binario si rodal i implementa politica j

### 4. Restricciones: (explicar en palabras y matematica)

1. **No mas que un manejo para cada rodal:**
   $$
   \sum_{j=1}^{M} X_{ij} <= 1 \quad \forall i
   $$

1. **Cumplir produccion minima anual:**

    $$
    \sum_{j=1}^{M} \sum_{i=1}^{R} A_{ijt}X_{ij} >= D \quad \forall t
    $$

1. **v[t]:**
    $$
    \sum_{j=1}^{M}\sum_{i=1}^{R} A_{ijt} X_{ij} =v_{t}
    $$

1. **Generacion estable de biomasa por periodo:**

    $$
    |v_{t} - v_{t-1}| <= \frac{v_{t-1}}{10}
    $$
1. **Gasto Menor al Presupuesto:**
    $$
    \sum_{j=1}^{M}\sum_{i=1}^{R}C_{ij}*X_{ij} <= B
    $$

### 5. Funcion objetivo

$$
\max_{x_{ij}\in\{0,1\}} (\sum_{i=1}^{R} \sum_{j=1}^{M}X_{ij} \sum_{t=1}^{H} \frac{A_{ijt}pm_i}{(1+tasa)^t})
$$

biom raleo = biom funcion pre raleo - biom funcion post raleo

#### Si se quema

- Se puede plantar al año siguiente o se espera mas años?
- se puede agregar restriccion de maximo cantidad de combustibles graves
- se puede quemar una parte del rodal pero no todo? (simulación de rodal)

hacer opti, simular, luego opti denuevo
hacer simulacion biomasa separado incendio
no solo el revenue,

### Modelo simple para prueba 1

#### Cambios

- b2m; biom = madera
- $P_m$= 63000 para todo tipo de madera, despues escala de manera random con random walk
- $C_{ij}$ = 10
- B = 400 (40 rodales pueden recibir manejo)
- R = 55 (rodales)
- D > 0

### Proposicion cal_biomasa

- Revisar si edad de rodal es menor a año estable (cada rodal tiene año estable)
- Si lo es entonces divide en año estable y se multiplica por edad real (lineal)

Agregar que quede una cartidad de biomasa en pie (puede ser la misma que la inicial)



### Ideas

Elegir entre 5 opciones con simulador
poner un limite de combustibles "peligrosos"
poner que desde un porcentaje de quema si o si (70% es quema) para aumentar riesgo
preguntar felipe como es la simulacion, porque si depende de altura y edad, eso se asume con el mismo combustible
dar un riesgo promedio a cada tipo de combustible, lo que se agrega a la optimizacion (despues igual se simula para revisar si entre con riesgo promedio y sin)
intentar iterar con los nuevos datos de vendible (no se sabria cuando se demora o cuantas iteraciones se necesite) se podrian poner 2 opciones para parar las iteraciones, o que de el mismo resultado denuevo, o que se queme menos que un porcentaje
paper del profe de optimizar por periodo post simular incendio, pero en esta ocacion con el simulador de incendios

