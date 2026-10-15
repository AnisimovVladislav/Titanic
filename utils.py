import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import ParameterGrid

import config as cnf 

# Подготовка данных
def predobr(df_train, df_test, df_gender_submission):
    df_train_priz = df_train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    df_train_priz['Age'] = df_train_priz['Age'].fillna(df_train_priz['Age'].median())
    df_train_priz['Embarked'] = df_train_priz['Embarked'].fillna('S')
    df_train_priz = pd.get_dummies(df_train_priz, dtype=float, drop_first=True)
    
    X = df_train_priz.drop('Survived', axis=1)
    y = df_train_priz['Survived']
    return X, y

# Обучение модели
def fit_model(model, X, y, train_index, test_index, scaler=False):
    current_X_train = X.iloc[train_index]
    current_y_train = y.iloc[train_index]
    current_X_test  = X.iloc[test_index]
    current_y_test  = y.iloc[test_index]
    
    if scaler:
        scaler  = StandardScaler()
        scaler.fit(current_X_train)
        current_X_train = scaler.transform(current_X_train)
        current_X_test = scaler.transform(current_X_test)
    
    model.fit(current_X_train, current_y_train)
    train_accuracy = model.score(current_X_train, current_y_train, sample_weight=None)
    test_accuracy = model.score(current_X_test, current_y_test, sample_weight=None)
    
    return train_accuracy, test_accuracy

# Получение средней accuracy и стандартного отклонения
def mean_accuracy(array_train_accuracy, array_test_accuracy):
        mean_train_accuracy = float(np.mean(array_train_accuracy))
        mean_train_std = np.std(array_train_accuracy)

        mean_test_accuracy = float(np.mean(array_test_accuracy))
        mean_test_std = np.std(array_test_accuracy)
        print("Train_accuracy: ", mean_train_accuracy, "Test_accuracy: ", mean_test_accuracy)
        
        return mean_train_accuracy, mean_train_std, mean_test_accuracy, mean_test_std
    
# Перебор параметров 
def parametrs(params):
    param = params.copy()
    for key, value in param.items():   
        if type(value) != list:
            param[key] = [value]

    return list(ParameterGrid(param))
