
# Clasificación de Ingresos (>50K)

Este proyecto tiene como objetivo predecir si un individuo gana más o menos de 50.000 unidades monetarias, a partir de características sociodemográficas como edad, educación, horas trabajadas, etc.

## Estructura del proyecto

├── data/                # Datos originales (CSV)
├── eda/                 # Notebook o script con el análisis exploratorio
├── preprocessing/       # Preprocesamiento y exportación de datos transformados
├── models/              # Modelos entrenados y archivos SHAP
├── utils.py             # Funciones auxiliares para el análisis
├── model_selection.py   # Entrenamiento y evaluación de modelos
├── conclusiones.md      # Hallazgos analíticos del proyecto
├── requirements.txt     # Dependencias del entorno
├── README.md            # Guía del proyecto
```

## Requisitos

Instalar las dependencias necesarias usando requirements.txt
```

## Instrucciones de ejecución

1. **EDA**: Ejecutar `eda.py` para realizar el análisis exploratorio de datos.
2. **Preprocesamiento**: Ejecutar `preprocessing.py` para transformar y guardar los datos.
3. **Modelado**: Ejecutar `model_selection.py` para entrenar, evaluar y guardar los modelos.
4. **Conclusiones**: Leer las conclusiones para observar los hallazgos en profundidad.