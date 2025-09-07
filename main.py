# main.py
# Proyecto: Clasificación de supervivencia en el Titanic usando Regresión Logística
# Descripción: Este script carga, preprocesa, entrena y evalúa un modelo de regresión logística sobre el dataset del Titanic.
# El algoritmo de regresión logística es un método de clasificación supervisada que estima la probabilidad de que una instancia pertenezca a una clase (por ejemplo, sobrevivir o no).
# Utiliza la función sigmoide para transformar una combinación lineal de las variables independientes en una probabilidad entre 0 y 1.
# El modelo se entrena ajustando los coeficientes para maximizar la verosimilitud de los datos observados.

from src.data_processing.data_loader import load_titanic_data  # Carga el dataset desde Kaggle
from src.data_processing.data_preprocessor import preprocess_data  # Preprocesa los datos (limpieza y codificación)
from src.models.logistic_regression import train_logistic_regression, save_model  # Entrena y guarda el modelo
from src.evaluation.metrics_calculator import calculate_metrics  # Calcula métricas de desempeño
from src.evaluation.model_evaluator import evaluate_model  # Evalúa el modelo sobre el conjunto de prueba
from src.visualization.data_visualizer import plot_correlation_matrix  # Grafica la matriz de correlación
from src.visualization.model_visualizer import plot_feature_importance  # Grafica la importancia de las variables
from src.utils.config import *  # Rutas de archivos y configuración
from sklearn.model_selection import train_test_split  # Divide los datos en entrenamiento y prueba
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    # 1. Cargar el dataset del Titanic
    df = load_titanic_data()  # Descarga y lee el archivo CSV

    # 2. Preprocesar los datos
    # - Elimina columnas irrelevantes
    # - Imputa valores faltantes
    # - Codifica variables categóricas
    df = preprocess_data(df)

    # 3. Analizar la correlación entre variables
    # - Genera y guarda una matriz de correlación para identificar relaciones y posibles variables relevantes
    plot_correlation_matrix(df, CORR_MATRIX_PATH)

    # 4. Separar variables independientes (X) y dependiente (y)
    # - X: Variables predictoras
    # - y: Variable objetivo ('Survived')
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 5. Dividir el dataset en entrenamiento y prueba
    # - Permite evaluar el modelo en datos no vistos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Entrenar el modelo de regresión logística
    # - Ajusta los coeficientes para predecir la probabilidad de supervivencia
    # - La función sigmoide transforma la salida en una probabilidad
    model = train_logistic_regression(X_train, y_train)

    # 7. Guardar el modelo entrenado
    save_model(model, MODEL_PATH)

    # 8. Evaluar el modelo
    # - Calcula métricas como accuracy, precisión, recall, F1 y matriz de confusión
    metrics = evaluate_model(model, X_test, y_test, calculate_metrics)
    print("Metrics:", metrics)

    # Extrae la matriz de confusión calculada previamente en las métricas del modelo. 'cm' es un arreglo 2x2 con los conteos de predicciones correctas e incorrectas.
    cm = metrics['confusion_matrix']
    # Crea una nueva figura de matplotlib con tamaño 6x5 pulgadas para la gráfica.
    plt.figure(figsize=(6,5))
    # Dibuja un mapa de calor (heatmap) de la matriz de confusión usando seaborn.
    # - 'annot=True' muestra los valores en cada celda.
    # - 'fmt="d"' muestra los valores como enteros.
    # - 'cmap="Blues"' usa una paleta azul.
    # - 'xticklabels' y 'yticklabels' ponen etiquetas descriptivas en los ejes.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No sobrevivió', 'Sobrevivió'], yticklabels=['No sobrevivió', 'Sobrevivió'])
    # Añade la etiqueta 'Predicción' al eje X.
    plt.xlabel('Predicción')
    # Añade la etiqueta 'Real' al eje Y.
    plt.ylabel('Real')
    # Añade el título a la gráfica.
    plt.title('Matriz de Confusión - Regresión Logística')
    # Ajusta el diseño para evitar que los elementos se solapen.
    plt.tight_layout()
    # Asegura que la carpeta 'results' exista para guardar la imagen.
    os.makedirs('results', exist_ok=True)
    # Guarda la gráfica como imagen PNG en la carpeta 'results'.
    plt.savefig('results/confusion_matrix.png')
    # Cierra la figura para liberar memoria.
    plt.close()

    # 9. Analizar la importancia de las variables
    # - Grafica los coeficientes absolutos del modelo para ver qué variables influyen más en la predicción
    plot_feature_importance(model, X.columns.tolist(), FEATURE_IMPORTANCE_PATH)

if __name__ == "__main__":
    main()
