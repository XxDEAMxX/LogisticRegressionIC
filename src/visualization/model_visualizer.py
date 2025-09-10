import matplotlib.pyplot as plt
import numpy as np
import os

def plot_feature_importance(model, feature_names, output_path):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Usar el valor absoluto de los coeficientes para mostrar la importancia
    importance = np.abs(model.coef_[0])
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(8,6))
    plt.bar([feature_names[i] for i in indices], importance[indices])
    plt.xticks(rotation=45)
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.ylabel('Importancia (valor absoluto del coeficiente)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Calcular importancia en porcentaje
    importance_percent = 100 * importance / importance.sum()
    plt.figure(figsize=(8,6))
    plt.bar([feature_names[i] for i in indices], importance_percent[indices])
    plt.xticks(rotation=45)
    plt.title('Feature Importance (%)')
    plt.ylabel('Importancia (%)')
    plt.tight_layout()
    percent_path = os.path.join(os.path.dirname(output_path), 'feature_importance_percent.png')
    plt.savefig(percent_path)
    plt.close()
