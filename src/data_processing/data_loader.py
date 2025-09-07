import pandas as pd
import kagglehub
import os

def load_titanic_data():
    # Descarga el dataset y obtiene la ruta
    path = kagglehub.dataset_download("yasserh/titanic-dataset")
    print("Path to dataset files:", path)
    # Buscar el primer archivo CSV en la carpeta descargada
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_path = os.path.join(path, file)
            print(f"Usando archivo: {csv_path}")
            df = pd.read_csv(csv_path)
            return df
    raise FileNotFoundError("No se encontró ningún archivo CSV en la carpeta descargada.")
