import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_correlation_matrix(df, output_path):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()
