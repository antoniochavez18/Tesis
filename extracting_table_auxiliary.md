(dev) fdo@fdeb:~/source/crecimiento$ pip install PyPDF2
Collecting PyPDF2
  Downloading pypdf2-3.0.1-py3-none-any.whl.metadata (6.8 kB)
Downloading pypdf2-3.0.1-py3-none-any.whl (232 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 232.6/232.6 kB 4.1 MB/s eta 0:00:00
Installing collected packages: PyPDF2
Successfully installed PyPDF2-3.0.1
(dev) fdo@fdeb:~/source/crecimiento$ ipython
Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.14.0 -- An enhanced Interactive Python. Type '?' for help.

[ins] In [1]: import PyPDF2
         ...: 
         ...: # creating a pdf reader object
         ...: reader = PyPDF2.PdfReader('Modelos de predicción de biomasa a nivel de rodal en plantaciones de Eucalyptus globulus en Chile.pdf')
         ...: 
         ...: # print the number of pages in pdf file
         ...: print(len(reader.pages))
         ...: 
         ...: # print the text of the first page
19

[ins] In [2]: print(reader.pages[4].extract_text())
[ins] In [2]: print(reader.pages[5].extract_text())
[ins] In [2]: print(reader.pages[4].extract_text())
4 | Pá g i n a
 • Condición : Es la condición que se encuentra el rodal relativa al esquema de ma nejo .
Esta puede ser posterior a Raleo, o posterior a Poda y r aleo.

Resultados

Los resultados pueden ser observados en l os siguientes cuadros . Cabe recorda r que los
coeficientes expuestos en estos son los correspondientes al mode lo presentado en la
ecuación 1.

Cuadro 1: coeficientes de modelo no lineal para la estimación de biomasa según la edad de
rodal para Eucalyptus glo bulus, seg ún Zona, densidad inicial , índice de sitio, manejo y
condición.
Zona  Densidad Inicial  Índice de sitio (SI)  Manej
o Condi
ción  α β γ
1 800 24
NA Sin
Manejo  2.371  1.778  0.000
26 2.371  1.778  0.000
28 5.464  1.591  0.000
30 7.709  1.519  0.000
32 10.228  1.462  6.240
1250  24 2.379  1.774  0.000
26 3.794  1.666  0.000
28 5.307  1.600  0.000
30 7.779  1.513  0.000
32 10.304  1.457  5.660
2 800 24 3.221  1.647  0.000
26 3.612  1.656  0.000
28 4.559  1.618  0.000
30 5.926  1.567  0.000
32 7.300  1.532  0.000
1250  24 3.197  1.650  0.000
26 3.729  1.645  0.000
28 5.129  1.574  0.000
30 6.201  1.547  0.000
32 7.592  1.514  0.000






[ins] In [3]: print(reader.pages[5].extract_text())
5 | Pá g i n a
 Cuadro 2: coeficientes de modelo no lineal para la estimación de biomasa según la edad de
rodal para Pinus  radiata , seg ún Zona, densidad inicial , índice de sitio, manejo y condición.
Zona  Densidad
Inicial  Índice de
sitio (SI)  Manejo  Condición  α β γ
Z6
1250  32 Intensivo  Post Raleo 1250 -700 0.421  2.253  3.485
Z7 0.576  2.044  3.968
Z6 Post Poda y Raleo
700-300 122.491  0.562  -
416.814
Z7 22.934  0.936  -
171.991
Z6
29 Intensivo2  Post Raleo 1250 -700 0.421  2.253  3.485
Z7 0.070  2.850  7.522
Z6 Post Poda y Raleo
700-300 122.491  0.562  -
416.814
Z7 40.929  0.774  -
227.526
Z6
26 Multipropósito  Post Raleo 1250 -700 0.069  2.791  1.441
Z7 0.047  2.905  8.454
Z6 Post Poda y Raleo
700-300 58.378  0.701  -
298.450
Z7 56.816  0.666  -
260.576
Z6
23 Pulpable  Post Raleo 1250 -700 9.592  1.169  -97.246
Z7 20.852  0.844  -
103.893

Las curvas de crecimiento obtenidas pueden ser observadas en los anexos 1 y 2.  En estas,
las curvas generadas por los mod elos corresponde n a líneas  continuas de color rojo, y los
puntos asociados a los dat os real es están representados por círculos  azules.


