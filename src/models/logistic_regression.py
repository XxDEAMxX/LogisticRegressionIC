"""
logistic_regression.py

Este módulo implementa la lógica para entrenar, guardar y cargar un modelo de regresión logística usando scikit-learn.
La regresión logística es un algoritmo de clasificación supervisada que estima la probabilidad de que una instancia pertenezca a una clase (por ejemplo, sobrevivir o no en el Titanic).
Utiliza una función sigmoide para transformar una combinación lineal de las variables independientes en una probabilidad entre 0 y 1.
"""

from sklearn.linear_model import LogisticRegression
import pickle


def train_logistic_regression(X_train, y_train):
    """
    Entrena un modelo de regresión logística sobre los datos de entrenamiento.
    Args:
        X_train (DataFrame): Variables independientes (predictoras) del conjunto de entrenamiento.
        y_train (Series): Variable dependiente (objetivo) del conjunto de entrenamiento.
    Returns:
        model (LogisticRegression): Modelo entrenado de regresión logística.
    """
    # Crea una instancia del modelo de regresión logística con un máximo de 1000 iteraciones para asegurar la convergencia.
    model = LogisticRegression(max_iter=1000)
    # Entrena el modelo ajustando los coeficientes internos para encontrar la mejor relación entre las variables independientes (X_train)
    # y la variable dependiente (y_train). El modelo aprende a predecir la probabilidad de cada clase usando la función sigmoide.
    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    """
    Guarda el modelo entrenado en un archivo usando pickle.
    Args:
        model (LogisticRegression): Modelo entrenado.
        path (str): Ruta donde se guardará el archivo.
    """
    # Abre el archivo en modo escritura binaria ('wb') en la ruta especificada.
    with open(path, 'wb') as f:
        # Serializa (convierte en bytes) el modelo entrenado y lo guarda en el archivo usando pickle.
        pickle.dump(model, f)


def load_model(path):
    """
    Carga un modelo previamente guardado desde un archivo.
    Args:
        path (str): Ruta del archivo donde está guardado el modelo.
    Returns:
        model (LogisticRegression): Modelo cargado listo para hacer predicciones.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
