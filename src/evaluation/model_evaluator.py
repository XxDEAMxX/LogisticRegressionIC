def evaluate_model(model, X_test, y_test, metrics_calculator):
    # Utiliza el modelo entrenado para predecir las clases del conjunto de prueba (X_test).
    y_pred = model.predict(X_test)
    # Calcula las métricas de evaluación (accuracy, precisión, recall, F1, matriz de confusión, etc.)
    # usando la función metrics_calculator y las predicciones obtenidas.
    return metrics_calculator(y_test, y_pred)
