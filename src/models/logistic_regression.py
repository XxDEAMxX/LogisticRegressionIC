"""
logistic_regression.py

Este módulo implementa la lógica para entrenar, guardar y cargar un modelo de regresión logística usando scikit-learn.
La regresión logística es un algoritmo de clasificación supervisada que estima la probabilidad de que una instancia pertenezca a una clase (por ejemplo, sobrevivir o no en el Titanic).
Utiliza una función sigmoide para transformar una combinación lineal de las variables independientes en una probabilidad entre 0 y 1.
"""

from sklearn.linear_model import LogisticRegression
import pickle


def train_logistic_regression(X_train, y_train, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight=None):
    """
    Entrena un modelo de regresión logística con hiperparámetros ajustables.
    
    Args:
        X_train (DataFrame): Variables independientes del conjunto de entrenamiento.
        y_train (Series): Variable dependiente del conjunto de entrenamiento.
        C (float): Regularización inverso (menor = más regularización).
        penalty (str): Tipo de regularización ('l1', 'l2').
        solver (str): Algoritmo de optimización ('lbfgs', 'liblinear').
        max_iter (int): Iteraciones máximas para convergencia.
        class_weight (str): Manejo de clases desbalanceadas ('balanced').
    """
    # C=1.0: Balance estándar entre sesgo y varianza para datasets pequeños
    # penalty='l2': Regularización estable que evita overfitting sin eliminar variables
    # solver='lbfgs': Eficiente para datasets pequeños/medianos con regularización L2
    # max_iter=1000: Garantiza convergencia (usualmente converge en <100 iteraciones)
    # class_weight=None: Sin ajuste automático de pesos (usar 'balanced' si hay desbalance)
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight
    )
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
