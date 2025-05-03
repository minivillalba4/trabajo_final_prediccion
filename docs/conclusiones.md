# 1. Objetivo del Análisis

El presente análisis exploratorio de datos tiene como propósito comprender en profundidad las características y relaciones del dataset disponible con el fin de preparar el terreno para un modelo de predicción. El objetivo final es construir un modelo capaz de **predecir si un individuo recibe un salario superior a 50.000 unidades monetarias (u.m.)**, en función de diversas variables sociodemográficas y laborales.

Este análisis preliminar permite:

- Detectar y tratar valores atípicos y ausentes.
- Comprender la distribución de las variables y sus relaciones con la variable objetivo (`income`).
- Identificar posibles transformaciones o codificaciones necesarias antes de entrenar modelos.
- Obtener los primeros **insights de negocio** que ayuden a comprender los factores que más se asocian con ingresos altos.

El dataset contiene **48.842 registros** y **15 columnas**: 14 variables descriptivas y una variable objetivo binaria.


# 2. Diccionario de Datos

A continuación se describen las variables del dataset junto con su significado y tipo esperado:

| Variable         | Descripción                                                                 | Tipo esperado   |
|------------------|------------------------------------------------------------------------------|-----------------|
| `age`            | Edad del individuo                                                           | Numérica (`int`) |
| `workclass`      | Tipo de empleador del individuo (por ejemplo: `Private`, `Self-emp`, etc.) | Categórica       |
| `fnlwgt`         | Peso final, usado como estimador para representar a la población general     | Numérica (`int`) |
| `education`      | Nivel educativo alcanzado (por ejemplo: `Bachelors`, `HS-grad`, etc.)        | Categórica       |
| `education_num`  | Representación numérica del nivel educativo                                  | Numérica (`int`) |
| `marital_status` | Estado civil del individuo                                                   | Categórica       |
| `occupation`     | Tipo de ocupación o trabajo                                                  | Categórica       |
| `relationship`   | Relación familiar con respecto al hogar                                      | Categórica       |
| `race`           | Raza o grupo étnico declarado                                                | Categórica       |
| `sex`            | Sexo biológico del individuo (`Male`, `Female`)                              | Categórica       |
| `capital_gain`   | Ganancias de capital obtenidas                                               | Numérica (`int`) |
| `capital_loss`   | Pérdidas de capital reportadas                                               | Numérica (`int`) |
| `hours_per_week` | Número de horas trabajadas por semana                                        | Numérica (`int`) |
| `native_country` | País de origen del individuo                                                 | Categórica       |
| `income`         | Variable objetivo: `>50K` si gana más de 50.000 u.m., `<=50K` si no          | Categórica binaria (`0/1`) |

Este diccionario servirá como base para estructurar el EDA, validar los tipos de datos reales frente a los esperados, y guiar la limpieza, transformación y análisis de cada variable.

# Inspección inicial del conjunto de datos
## 3. Incongruencias y validación de tipos

Tras la inspección inicial del `DataFrame` mediante el método `df.info()`, se ha verificado que el dataset cuenta con 15 columnas, de las cuales 6 son de tipo `int64` y 9 son `object`. Aunque esto podría parecer correcto a primera vista, al contrastarlo con el diccionario de datos se han detectado varias **incongruencias y aspectos a mejorar** para asegurar la correcta interpretación y tratamiento de las variables:

1. **Valores faltantes codificados como "?"**  
   Algunas variables categóricas contienen el valor `"?"` para representar datos ausentes. Esto no se detecta como `NaN` por Pandas, lo cual puede enmascarar el análisis de valores nulos y afectar la limpieza posterior. Será necesario reemplazar todos los `"?"` por `np.nan`.

2. **Inconsistencia entre los nombres del dataset y el diccionario**  
   La variable `gender` aparece en el dataset, mientras que el diccionario se refiere a esta como `sex`. Para mantener consistencia y trazabilidad, se recomienda renombrarla a `sex`, como indica el diccionario.


Estas correcciones permitirán un EDA más limpio, coherente y riguroso, además de evitar errores en fases posteriores del proyecto (preprocesado, modelado y visualización).

## 4. Valores nulos y duplicados

### Valores duplicados

Se identificaron **52 registros duplicados exactos** (≈ 0.11 % del total). Dado que el dataset no contiene identificadores únicos, es razonable asumir que dichos duplicados reflejan perfiles repetidos legítimos. Por tanto, **no se han eliminado**, ya que su presencia mantiene la representatividad estadística del conjunto.

### Valores nulos

Tras sustituir los valores `"?"` por `NaN`, se detectaron nulos en tres columnas categóricas:

| Variable         | Valores nulos |
|------------------|----------------|
| `workclass`      | 2 799          |
| `occupation`     | 2 809          |
| `native_country` | 857            |

En lugar de imputar por la moda, se optó por **crear una categoría explícita** llamada `"Desconocido"` para conservar la información sobre la ausencia de datos sin introducir supuestos. Esta técnica permite que los modelos aprendan a distinguir también los casos con información faltante, lo cual puede aportar valor predictivo.


# Analisis Univariable
El análisis univariable tiene como objetivo explorar cada variable categórica de forma individual, identificando su distribución de frecuencias y su nivel de concentración o dispersión. Esta fase permite detectar valores dominantes, categorías residuales y posibles inconsistencias en los datos, sirviendo como base para futuras decisiones de limpieza, transformación y codificación.

## 8. Análisis de Variables Categóricas
Durante el análisis univariable de las variables numéricas, se observó que tanto capital-gain como capital-loss presentaban una alta concentración de ceros, con más del 90 % de los registros sin ganancias ni pérdidas de capital. Esta distribución extremadamente sesgada quedó evidenciada en los histogramas, donde la mayoría de los valores se agrupaban en torno al cero y apenas un pequeño grupo mostraba variabilidad. Dado que esta falta de dispersión limita la capacidad predictiva directa de estas variables en su formato original, se optó por llevar a cabo un enfoque iterativo y transformarlas en variables booleanas (has_capital_gain y has_capital_loss) que indican simplemente la presencia o ausencia de ganancia o pérdida de capital, respectivamente. Esta transformación permite conservar la señal relevante sin introducir ruido de manera innecesaria.

### `workclass`

El 69.4 % trabaja en el sector privado, mientras que los sectores autónomos (11.4 % entre `Self-emp-not-inc` y `Self-emp-inc`) y públicos (13.4 % entre `Local-gov`, `State-gov` y `Federal-gov`) también tienen una presencia destacable. Se ha detectado un 5.7 % de registros con "desconocido".
Se observa un fuerte predominio del sector privado (69.4 %), mientras que los trabajadores del sector público y autónomos representan una proporción significativamente menor. Este desbalance puede hacer que el modelo tienda a aprender patrones sobreempleo asociados a este grupo mayoritario, y no generalice bien sobre colectivos menos representados. Se recomienda evaluar si es útil agrupar las clases en privado, público y autónomo.

### `education`

Los niveles `HS-grad` (32.3 %), `Some-college` (22.3 %) y `Bachelors` (16.4 %) concentran más del 70 % de la muestra. Las categorías con menor nivel educativo, como `Preschool` (0.2 %) o `1st-4th` (0.5 %), tienen una presencia residual. Dado que esta variable representa el nivel educativo alcanzado (categórica nominal), se mantendrá su granularidad para conservar la riqueza informativa.

### `marital-status`

El 45.8 % está casado legalmente, el 33.0 % nunca se ha casado y un 13.6 % está divorciado. El campo indica el estado civil del individuo. Las categorías menos frecuentes, como `Married-spouse-absent` (1.3 %) o `Married-AF-spouse` (0.08 %), podrían agruparse si fuera necesario.

### `occupation`

Las ocupaciones más comunes son `Prof-specialty` (12.6 %), `Craft-repair` (12.5 %), `Exec-managerial` (12.5 %) y `Adm-clerical` (11.5 %). También hay un 5.7 % de registros con valor "desconocido". Esta variable describe el tipo de ocupación habitual y será conservada tal como está. Se podrá valorar una agrupación temática posterior.

### `relationship`

`Husband` (40.4 %) y `Not-in-family` (25.8 %) son las relaciones predominantes, seguidas por `Own-child` (15.5 %), `Unmarried` (10.5 %), `Wife` (4.8 %) y `Other-relative` (3.1 %). Esta variable describe la relación del individuo con el cabeza de hogar y permite inferir la estructura familiar. Se mantendrá sin modificaciones.

### `race`

White representa el 85.5 %, seguido de Black (9.6 %) y otros grupos minoritarios. Aunque la raza no debería ser un factor determinante en los ingresos desde un punto de vista biológico, en la realidad socioeconómica que representa este dataset podrían existir correlaciones estructurales (desigualdades históricas, acceso a oportunidades, etc.). Este desbalance podría dificultar que el modelo aprenda patrones fiables para grupos étnicos minoritarios, por lo que se recomienda tratar esta variable con especial cautela para evitar sesgos o interpretaciones erróneas.

### `sex`

Aunque no es una variable de alta cardinalidad, existe un desbalance de género (66.9 % hombres). Esto puede tener también implicaciones éticas y predictivas.

### `native-country`

El 89.7 % es de EE.UU. El resto de países aparece con una frecuencia muy baja (<2 %). Este campo representa el país de origen declarado. Dado el claro dominio de una única categoría, se propone recodificar esta variable como un booleano: `es_estadounidense` vs. `no_lo_es`, lo que puede facilitar su interpretación y evitar que el modelo aprenda ruido de categorías poco representadas.

### `income` (variable objetivo)

El 76.1 % de los individuos gana `<=50K`, frente al 23.9 % que obtiene mayores ingresos. Aunque este reparto muestra cierto desbalance, **no es lo suficientemente extremo como para requerir técnicas agresivas de balanceo** como SMOTE.

No obstante, para evitar que los modelos infravaloren la clase minoritaria, se aplicará la estrategia de **ponderación de clases automática** mediante el parámetro `class_weight='balanced'` en los algoritmos que lo admitan (como `LogisticRegression`, `RandomForestClassifier` o `SVC`). Esta técnica ajusta internamente el peso de cada clase en función de su frecuencia, lo que permite **mejorar la sensibilidad del modelo sin necesidad de alterar el dataset original**. Se recomienda monitorizar en todo caso las métricas específicas por clase para garantizar un buen rendimiento en ambas categorías.


### `has_capital_gain` y `has_capital_loss`

Tras su transformación en variables binarias, se observa que solo el 8.3 % de los individuos han declarado ganancias de capital (`has_capital_gain = True`) y apenas un 4.7 % han declarado pérdidas (`has_capital_loss = True`). Este resultado confirma lo anticipado en el análisis univariable de las variables originales: la mayoría de los registros tienen valor cero.

Estas nuevas variables permiten capturar una señal informativa clave la existencia o no de ingresos pérdidas no salariales sin que los modelos se vean afectados por la dispersión o los outliers presentes en los valores originales.


## Análisis Univariable de Variables Numéricas

### `age`

La variable presenta una distribución sesgada hacia edades más jóvenes, con una concentración entre los 20 y 40 años. En el boxplot original aparecen valores a partir de los 70–75 años, aunque no se consideran erróneos ni improbables. Sin embargo, **al aplicar la transformación logarítmica desaparecen estos valores atípicos en el boxplot**, lo que sugiere que dicha transformación **puede ser útil para reducir la influencia de extremos sin eliminar información relevante**. Se recomienda conservar esta opción para etapas posteriores del modelado.

### `fnlwgt`

Para esta característica, se recomienda evaluar su **importancia en el modelo** (feature importance). De esta manera, sabremos si aporta señal predictiva o introduce ruido. Si `fnlwgt` muestra baja relevancia, se podrá eliminar en fases posteriores. En caso contrario, podría mantenerse o transformarse para ajustar su escala.

**La aplicación de una transformación logarítmica ha resultado positiva**, ya que la distribución inicial mostraba una fuerte asimetría y una gran cantidad de valores extremos. Tras el cambio de escala, la variable se aproxima a una distribución más simétrica y controlada, lo que podría favorecer el rendimiento de algunos modelos sensibles a escalas o outliers.

### `educational-num`

Como variable ordinal que codifica el nivel educativo (1 a 16), su distribución muestra picos en valores intermedios que, en datasets similares, suelen corresponder (de acuerdo a búsquedas externas) a niveles como `Some-college`, `HS-grad` o `Bachelors`. Sin embargo, dado que el mapeo exacto de `education_num` no ha sido verificado sobre este dataset, **no se puede afirmar con certeza** qué valor numérico corresponde a cada nivel educativo. En cualquier caso, se trata de una variable bien estructurada y no requiere transformación. Puede utilizarse directamente como variable numérica.

### `hours-per-week`

La mayoría de las personas declara **exactamente 40 horas semanales**; los valores por debajo de 10 h o por encima de 80 h son muy raros. Probar la escala logarítmica no mejora la visualización: el pico principal solo se desplaza y la variable pierde su significado intuitivo. Por tanto, es mejor dejarla tal cual.

Respecto a los valores extremos, aunque son infrecuentes, no se consideran errores sino casos reales de jornadas atípicas.


# Analisis Bivariable
## Variables Categóricas

### `education`

Existe una relación creciente entre el nivel educativo y los ingresos altos. Las personas con Doctorado, Masters o Bachelors tienen mayor probabilidad de superar los 50K (hasta un 72.6 % en Doctorado). En cambio, los niveles bajos como Preschool, 1st–4th, 5th–6th presentan tasas residuales.

Test chi-cuadrado: p < 0.001. Asociación altamente significativa** entre educación e ingresos.

### `has_capital_gain` y `has_capital_loss`

Las variables binarias derivadas de las ganancias y pérdidas de capital muestran una fuerte asociación con los ingresos altos. En concreto, el **61.7 % de quienes declaran alguna ganancia de capital** (has_capital_gain = True) superan los 50K, frente a solo un 20.5 % de quienes no tienen ganancias. En cuanto a las pérdidas, el **50.1 % de quienes declaran pérdidas de capital** (has_capital_loss = True) también ganan más de 50K, frente al 22.6 % del resto.

Test chi-cuadrado: p < 0.001. Asociación estadísticamente muy significativa con la variable objetivo.

### `marital-status`

Los **casados legalmente** (`Married-civ-spouse`) concentran el 46.6 % de ingresos altos. En cambio, los **nunca casados** o separados presentan proporciones muy inferiores (<10 %).

**Conclusión**: El estado civil tiene **una asociación clara con los ingresos**, posiblemente porque la una estabilidad familiar les ayude a conseguir una mejor situación laboral.

Test chi-cuadrado: p < 0.001. Asociación muy significativa.

### `native-country`

Como se ha mencionado anteriormente, 89.7 % de los individuos del dataset proviene de EE.UU., mientras que el resto de países están representados de forma muy minoritaria, lo que **limita la fiabilidad estadística** de cualquier patrón observado fuera de ese grupo. Si bien en algunas regiones como Asia o Europa se aprecian ligeras variaciones en la proporción de ingresos altos, la muestra es demasiado reducida como para extraer conclusiones sólidas.

Test chi-cuadrado: p < 0.001. Aunque el test sugiere una asociación significativa, la baja representación en muchas categorías podría sesgar el resultado.

Por ello, re recomienda **recodificar esta variable como una binaria**: `es_estadounidense` vs. `no_lo_es`, para facilitar la interpretación, evitar ruido por escasez de datos en categorías minoritarias, y mantener una señal informativa manejable en el modelo.

### `occupation`

Ocupaciones como `Exec-managerial`, `Prof-specialty` y `Sales` muestran los mayores porcentajes de ingresos altos. Aunque otras profesiones relacionadas con las fuerzas armadas, la protección y la seguridad representan también un gran porcentaje, no tienen un conteo de casos lo suficientemente alto como para ser lo suficientemente confiables.

Test chi-cuadrado: p < 0.001. Asociación clara y significativa con el nivel de ingresos.

### `race`

El 85.5 % de los registros se identifica como **White**, mientras que el resto de grupos étnicos apenas supera el 9.6 % (`Black`), 3.1 % (`Asian-Pac-Islander`) o menos del 1 % (`Amer-Indian-Eskimo`, `Other`). Este **desequilibrio extremo** implica que los patrones observados para las minorías carecen de **robustez estadística** y pueden estar sesgados por su baja representación.

Aunque existen diferencias en la proporción de ingresos altos (por ejemplo, 26.9 % para `Asian-Pac-Islander` vs. 25.4 % para `White`), la muestra de las categorías minoritarias es tan pequeña que no permite inferencias confiables.

Test chi-cuadrado: p < 0.001. Asociación estadísticamente significativa, aunque posiblemente influida por desbalance muestral.

### `relationship`

En términos de ingresos, Husband y Wife concentran la mayor proporción de ingresos >50K (ambos por encima del 44 %), mientras que el resto de relaciones apenas superan el 10 %.

Esta variable ofrece una señal clara sobre la estructura y rol familiar del individuo, con impacto en los ingresos. Sin embargo, hay que tener en cuenta las **diferencias notables en tamaño muestral entre categorías**, existiendo una mayor recolección de observaciones en Husband que en Wife

Test chi-cuadrado: p < 0.001 (*). Relación estadísticamente significativa con la variable objetivo.

### `sex`

Los **hombres (30.4 %)** tienen casi el triple de probabilidad de ganar más de 50K respecto a las mujeres (10.9 %).

Test chi-cuadrado: p < 0.001 (***). Asociación estadísticamente significativa entre sexo e ingresos. Sin embargo, esto no implica causalidad directa: esta diferencia puede deberse a múltiples factores estructurales (ocupación, nivel educativo, carga familiar, etc.). Para comprender el origen real de esta desigualdad sería necesario un análisis más profundo o estudios complementarios.

### `workclass`

Se ha llevado a cabo una agrupación temática con el fin de simplificar la informcación.  
Tras reagrupar los distintos tipos de empleo en cinco categorías (privado, público, autónomo, desconocido y sin empleo), se observa que los autónomos (36.3 %) y los empleados del sector público (30.8 %) presentan las mayores proporciones de ingresos superiores a 50K. En contraste, el sector privado, aunque es claramente el más numeroso, muestra una tasa inferior (21.8 %). Las categorías desconocido (9.5 %) y sin empleo (6.5 %) tienen una probabilidad significativamente menor de superar ese umbral.

Test chi-cuadrado: p < 0.001. Asociación altamente significativa.


## Variables Numéricas

### `age`

El histograma por clase muestra que la distribución de edades para quienes ganan más de 50K se desplaza hacia valores mayores, con un pico claro entre los **40 y 55 años**, mientras que la clase `<=50K` se concentra en edades más jóvenes (20 a 35 años). Esto se confirma estadísticamente con un p-valor < 0.001, altamente significativo.

Por ello, la variable edad **tiene una fuerte asociación con los ingresos**, siendo un posible predictor la experiencia laboral de cada persona.

### `fnlwgt`

Aunque se aprecia una ligera diferencia visual entre clases, el **test arroja un p-valor de 0.153 (no significativo)**. Esto indica que la diferencia de medias no es estadísticamente relevante.

fnlwgt no parece tener una relación significativa con la variable objetivo, por lo que habría que evaluar su utilidad más adelante con análisis de importancia de características.

### `educational-num`

Se observa una clara separación entre clases: la clase `>50K` está sobrerrepresentada en los valores altos de la variable, mientras que los niveles bajos son dominados por la clase `<=50K`. El p-valor < 0.001 indica alta significancia estadística**. Por ello, existe una **relación creciente entre el tipo de educación y probabilidad de altos ingresos

### `hours-per-week`

Ambas clases tienen un pico en torno a las 40 horas semanales (probablemente debido a la jornada completa), pero los ingresos `>50K` se concentran más en **valores superiores a las 45-50 horas**, mientras que la clase `<=50K` está más concentrada en valores inferiores. El p-valor < 0.001 valida la significancia estadística** de la diferencia.

Trabajar más horas está asociado a mayores ingresos, pues la mayoría de personas que pertenecen a la clase `>50K` se encuentran en un valor igual o superior a las 40 horas semanales.



# Análisis Multivariante
## Análisis de Correlaciones y Multicolinealidad

Se ha evaluado la correlación entre variables numéricas utilizando el coeficiente de **Spearman**y **Pearson**. Todas las correlaciones observadas son **bajas y no redundantes**, por lo que no hay indicios visibiles de multicolinealidad.


En este dataset, los valores más altos encontrados rondan el 0.4–0.65, lo que sugiere **algunas asociaciones relevantes que podrían derivar en redundancia**, pero **no necesariamente problemáticas** si se manejan adecuadamente en fases posteriores.

## V de Cramer
Los valores de V de Cramér observados (entre 0.35 y 0.65) **no son tan altos como para justificar la eliminación automática** de ninguna variable. Indican asociaciones **esperables y comprensibles**, algunas incluso deseadas por su capacidad predictiva. Se recomienda mantener todas las variables por el momento y abordar posibles redundancias o sesgos en la fase de modelado mediante **regularización, selección de variables o análisis SHAP**.

**1. Alta asociación entre** `relationship`, `sex` y `marital-status`  
V de Cramér entre 0.46 y 0.65  
Estas variables están **fuertemente relacionadas conceptualmente**. Por ejemplo, `Husband` implica simultáneamente sexo masculino y estado civil `Married-civ-spouse`. Esto puede generar redundancia en el modelo.

Evaluar en fases posteriores mediante técnicas de explicabilidad como SHAP o importancia de características. Si una de ellas no aporta información nueva, podría eliminarse.

**2. `occupation` y `sex` (V = 0.42)**  
La fuerte asociación entre estas variables sugiere que ambas están codificando información similar. Esta relación podría deberse a múltiples factores. Con los datos actuales no puede determinarse la causa de esta asociación. Se recomienda **vigilar su impacto en términos de equidad**.

**3. `relationship` e `income` (V = 0.45), `marital-status` e `income` (V = 0.45)**  
Ambas muestran una **asociación moderada con la variable objetivo**, lo que indica que podrían ser **predictoras útiles del nivel de ingresos**. Se justifica su conservación.

**4. `workclass` y `occupation` (V = 0.40)**  
La relación es **esperable y coherente**: una variable describe el tipo de relación contractual (`workclass`) y la otra el tipo de tarea (`occupation`). Aunque hay solapamiento, ambas pueden capturar matices diferentes del entorno laboral.

**5. `race` y `native-country` (V = 0.40)**  
Asociación moderada, comprensible por razones sociogeográficas. Si el modelo no muestra dependencia de ambas, podría considerarse priorizar `native-country` como binaria (`EE.UU.` vs `resto`) y monitorear `race`.

# **Preprocesamiento + Modelado**

- Se limpiaron valores "?" y se imputaron como `"desconocido"` (`workclass`, `occupation`, `native-country`).
- Se renombró `gender` a `sex` para unificar nombres.
- Se crearon variables binarias: `has_capital_gain`, `has_capital_loss`, `es_estadounidense`.
- Se codificó la variable objetivo `income` como 0/1.
- Se dividieron los datos (`train_test_split`) antes de cualquier transformación para evitar data leakage.
- Se aplicaron transformaciones:
  - `age` y `fnlwgt`: winsorización + logaritmo + escalado (para hacer que sea menos sensible a valores atípicos)
  - A otras variables solo se les aplicó escalado y winsorizacion por el mismo motivo (se comprobó que no era efectivo un tratamiento tan profundo con las mismas).
  - Ordinales: es decir,variables que siguen un cierto orden se les ha aplicado `OrdinalEncoder`.
  - Categóricas con mucha cardinalidad: se usó `TargetEncoder`.
- Todo el pipeline fue guardado y aplicado únicamente sobre train y posteriormente sobre test con el fin de evitar el data sampling bias

# **Modelado y evaluación**

- Se entrenaron 4 modelos con `class_weight='balanced'` para corregir el desbalance de clases:
  - Regresión logística, Random Forest, LightGBM, XGBoost.
- Se evaluaron con validación cruzada (3 folds) usando `accuracy`, `precision`, `recall` y `f1-score`.Tras comparar métricas, se seleccionaron los tres mejores modelos, ajustando hiperparámetros con **optimización bayesiana**
- Se construyó un modelo final tipo `VotingClassifier` (ensamblado soft) combinando Random Forest, LightGBM y XGBoost.
- El modelo ensamblado superó a todos los individuales en métricas F1 promedio (CV).
- Se aplicaron posteriores tecnicas de explicabilidad al modelo.

## Evaluación del modelo y análisis de métricas

### Contexto del problema  
En este conjunto de datos, la mayoría de personas (≈ 76 %) gana menos de 50.000, y solo una minoría (≈ 24 %) supera ese umbral. Por tanto, un modelo que simplemente predijera siempre "menos de 50K" tendría una **accuracy alta, pero no aportaría ningún valor real** para el objetivo que perseguimos: identificar correctamente a quienes sí tienen ingresos elevados.

### ¿Qué queríamos lograr?
En lugar de centrarnos en la precisión global (**accuracy**), nos propusimos **detectar bien a quienes sí ganan más de 50.000**, incluso si eso implicaba cometer algunos errores con el resto.
### ¿Qué consiguió el modelo?

| Métrica      | Valor | Interpretación                                                                 |
|--------------|--------|--------------------------------------------------------------------------------|
| **Recall**   | 0.76   | De cada 100 personas que **realmente ganan más de 50.000**, el modelo detecta 76. |
| **Precision**| 0.62   | De cada 100 personas que el modelo predice como **que ganan >50K**, acierta en 62. |
| **F1-score** | 0.69   | Resume el equilibrio entre precisión y recall. Buen compromiso global.        |
| **Accuracy** | 0.84   | Aunque alto, **no es la métrica más relevante** dado el desbalance de clases. |

El modelo **detecta bastante bien a las personas con ingresos altos** (que son pocas), aunque eso implique **fallar algunas veces con personas que en realidad no los tienen**. Esto es justo lo que se buscaba con la creación de este modelo.
