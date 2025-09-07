def evaluate_model(model, X_test, y_test, metrics_calculator):
    y_pred = model.predict(X_test)
    return metrics_calculator(y_test, y_pred)
