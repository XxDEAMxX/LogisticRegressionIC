from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """
    Calcula métricas derivadas de la matriz de confusión, incluyendo porcentajes y errores.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo

    Returns:
        Diccionario con matriz de confusión, porcentajes y análisis de errores
    """
    cm = confusion_matrix(y_true, y_pred)
    total_samples = cm.sum()
    correct_predictions = np.trace(cm)
    error_rate = (total_samples - correct_predictions) / total_samples
    cm_percent = cm / total_samples  # Matriz en porcentaje sobre el total
    return {
        "confusion_matrix": cm,
        "confusion_matrix_percent": cm_percent,
        "total_samples": int(total_samples),
        "correct_predictions": int(correct_predictions),
        "incorrect_predictions": int(total_samples - correct_predictions),
        "error_rate": float(error_rate)
    }
