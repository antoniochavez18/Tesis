# Instancia de ejemplo que muestra capacidades del simulador

6 casos: raleo despues cosecha, solo cosecha, solo raleo, cosecha después raleo, nada, dos periodos de cosecha y raleo
rellenando lo siguiente en el config_test.toml:
Seed (del random) 4, 10 rodales y 12 horizonte:
|Periodo   |R0_c18_r6  | R1  | R3_c10  |  R7_r8   |R8_c18_r10 |
|---|---------|-----|------------|------|-|
| 1 |   |    |    |    |    |
| 2 |   |    |    |    |    |
| 3 | c |    |    |    |    |
| 4 |   |    |    |    | r  |
| 5 |   |    |    |    |    |
| 6 |   |    | c  |    |    |
| 7 |   |    |    |    |    |
| 8 |   |    |    | r  |    |
| 9 | r |    |    |    |    |
|10 |   |    |    |    |    |
|11 |   |    |    |    |    |
|12 |   |    |    |    | c  |

Segundo caso
rellenando lo siguiente en el config_test.toml:
Seed (random) 4, 10 rodales y horizonte igual a 30

|Periodo   |R0_c18_r6  | R1  | R3_c10  |R8_c18_r10 |
|---|---------|-----|------------|-------|
| 1 |   |    |    |    |
| 2 |   |    |    |    |
| 3 | c |    |    |    |
| 4 |   |    |    | r  |
| 5 |   |    |    |    |
| 6 |   |    | c  |    |
| 7 |   |    |    |    |
| 8 |   |    |    |    |
| 9 | r |    |    |    |
|10 |   |    |    |    |
|11 |   |    |    |    |
|12 |   |    |    | c  |
|13 |   |    |    |    |
|14 |   |    |    |    |
|15 |   |    |    |    |
|16 |   |    | c  |    |
|17 |   |    |    |    |
|18 |   |    |    |    |
|19 |   |    |    |    |
|20 |   |    |    |    |
|21 | c |    |    |    |
|22 |   |    |    | r  |
|23 |   |    |    |    |
|24 |   |    |    |    |
|25 |   |    |    |    |
|26 |   |    |  c |    |
|27 | r |    |    |    |
|28 |   |    |    |    |
|29 |   |    |    |    |
|30 |   |    |    | c  |
