# Proyecto: Clasificación de Supervivencia en el Titanic usando Regresión Logística

Este proyecto implementa un modelo de regresión logística para predecir la supervivencia de los pasajeros del Titanic utilizando el dataset oficial de Kaggle. El código está organizado de forma modular para facilitar su comprensión y extensión.

## Estructura del Proyecto

```
RegLog/
├── requirements.txt
├── .gitignore
├── README.md
├── main.py
└── src/
    ├── data_processing/
    │   ├── data_loader.py
    │   └── data_preprocessor.py
    ├── models/
    │   └── logistic_regression.py
    ├── evaluation/
    │   ├── metrics_calculator.py
    │   └── model_evaluator.py
    ├── visualization/
    │   ├── data_visualizer.py
    │   └── model_visualizer.py
    └── utils/
        ├── config.py
        └── logger.py
```

## Requisitos

- Python 3.8 o superior
- Acceso a internet para descargar el dataset desde Kaggle

## Instalación de dependencias

Abre una terminal en la carpeta raíz del proyecto y ejecuta:

```pwsh
pip install -r requirements.txt
```

## Ejecución del proyecto

1. **Descarga automática del dataset:**
   El script descarga el dataset del Titanic desde Kaggle usando `kagglehub`. No necesitas descargarlo manualmente.

2. **Ejecuta el script principal:**

```pwsh
python main.py
```

Esto realizará los siguientes pasos:
- Carga y preprocesa los datos
- Analiza la correlación entre variables
- Entrena el modelo de regresión logística
- Evalúa el modelo y muestra métricas
- Genera gráficas de correlación y de importancia de variables en la carpeta `results/`

## Resultados

- Las métricas de evaluación se mostrarán en la terminal.
- Las gráficas y el modelo entrenado se guardarán en la carpeta `results/`.

## Notas sobre el algoritmo

La regresión logística es un método de clasificación supervisada que estima la probabilidad de que una instancia pertenezca a una clase (por ejemplo, sobrevivir o no). Utiliza la función sigmoide para transformar una combinación lineal de las variables independientes en una probabilidad entre 0 y 1.

## Personalización

Puedes modificar los scripts en `src/` para experimentar con diferentes variables, técnicas de preprocesamiento o métricas.

## Contacto

Para dudas o mejoras, contacta al autor o abre un issue en el repositorio.
