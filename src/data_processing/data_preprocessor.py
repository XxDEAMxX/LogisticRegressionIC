import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Eliminar columnas irrelevantes
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # Imputar valores faltantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Codificar variables categóricas
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    # Normalizar variables numéricas
    numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
