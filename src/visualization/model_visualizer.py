import matplotlib.pyplot as plt
import numpy as np
import os

def plot_feature_importance(model, feature_names, output_path):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance = np.abs(model.coef_[0])
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(8,6))
    plt.bar([feature_names[i] for i in indices], importance[indices])
    plt.xticks(rotation=45)
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
