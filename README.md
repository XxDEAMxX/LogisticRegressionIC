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

## Instalación con Entorno Virtual

### Configuración del entorno virtual (venv)

1. **Navega al directorio del proyecto**
   ```
   cd RegLog
   ```

2. **Crea un entorno virtual**
   ```
   python -m venv venv
   ```

3. **Activa el entorno virtual en Windows (PowerShell)**
   ```
   .\venv\Scripts\Activate.ps1
   ```
   
   **Nota**: Si obtienes un error de permisos, ejecuta primero:
   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Verifica que el entorno esté activo**
   - Deberías ver `(venv)` al inicio de tu línea de comandos
   - Esto indica que el entorno virtual está funcionando correctamente

5. **Instala las dependencias**
   ```
   pip install -r requirements.txt
   ```

## Ejecución del proyecto

1. **Asegúrate de que el entorno virtual esté activado**
   - Deberías ver `(venv)` al inicio de tu terminal en PowerShell.

2. **Ejecuta el script principal**
   ```
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
- Se generarán archivos de análisis de la matriz de confusión.

## Desactivar el entorno virtual

Cuando termines de trabajar con el proyecto, desactiva el entorno virtual:

```
deactivate
```

El indicador `(venv)` desaparecerá de tu línea de comandos.

## Notas sobre el algoritmo

La regresión logística es un método de clasificación supervisada que estima la probabilidad de que una instancia pertenezca a una clase (por ejemplo, sobrevivir o no). Utiliza la función sigmoide para transformar una combinación lineal de las variables independientes en una probabilidad entre 0 y 1.

## Personalización

Puedes modificar los scripts en `src/` para experimentar con diferentes variables, técnicas de preprocesamiento o métricas.

## Solución de problemas

- **Error de permisos en PowerShell**: Ejecuta `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` antes de activar el entorno.
- **Error de dependencias**: Asegúrate de tener activado el entorno virtual antes de instalar las dependencias.

## Contacto

Para dudas o mejoras, contacta al autor o abre un issue en el repositorio.
