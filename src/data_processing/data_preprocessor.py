import pandas as pd

def preprocess_data(df):
    # Eliminar columnas irrelevantes
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # Imputar valores faltantes
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # Codificar variables categóricas
    # Sex se convierte en Sex_male (1 si es hombre, 0 si es mujer).
    # Embarked se convierte en Embarked_Q y Embarked_S (Cherbourg se representa cuando ambos son 0).
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df
