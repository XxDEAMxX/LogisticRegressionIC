import matplotlib.pyplot as plt      # Importa la librería para crear gráficos en Python.
import seaborn as sns                # Importa seaborn para gráficos estadísticos avanzados.
import os                            # Importa os para manejar rutas y carpetas.

def plot_correlation_matrix(df, output_path):
    """
    Genera y visualiza una matriz de correlación para todas las variables numéricas del DataFrame.

    ¿Qué hace?
    - Calcula la correlación entre todas las variables numéricas usando df.corr().
    - Crea una gráfica tipo heatmap (mapa de calor) con seaborn, donde cada celda muestra el grado de correlación entre dos variables.
    - Añade anotaciones con los valores de correlación y un título descriptivo.
    - Ajusta el diseño para que la gráfica no se solape.
    - Guarda la imagen en la ruta especificada (output_path), creando la carpeta si no existe.
    - Muestra la gráfica en pantalla para facilitar el análisis visual.
    - Cierra la figura para liberar memoria.

    ¿Para qué sirve?
    - Permite identificar relaciones lineales entre variables, detectar multicolinealidad y seleccionar variables relevantes para el modelo.

    Args:
        df (pd.DataFrame): DataFrame con los datos procesados.
        output_path (str): Ruta donde se guardará la imagen de la matriz de correlación.
    """
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Asegura que la carpeta donde se guardará la imagen exista.
    corr = df.corr()  # Calcula la matriz de correlación entre todas las variables numéricas del DataFrame.
    plt.figure(figsize=(10,8))  # Crea una nueva figura de tamaño 10x8 pulgadas para la gráfica.
    sns.heatmap(corr, annot=True, cmap='coolwarm')  # Dibuja un mapa de calor (heatmap) de la matriz de correlación, mostrando los valores en cada celda y usando la paleta 'coolwarm'.
    plt.title('Correlation Matrix')  # Añade el título 'Correlation Matrix' a la gráfica.
    plt.tight_layout()  # Ajusta el diseño para evitar que los elementos se solapen.
    plt.savefig(output_path)  # Guarda la gráfica como imagen en la ruta especificada.
    plt.close()  # Cierra la figura para liberar memoria.
