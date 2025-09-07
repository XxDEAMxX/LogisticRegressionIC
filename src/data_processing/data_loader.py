import pandas as pd
import kagglehub

def load_titanic_data():
    path = kagglehub.dataset_download("yasserh/titanic-dataset")
    df = pd.read_csv(f"{path}/train.csv")
    return df
