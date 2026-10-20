import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

import re
from pathlib import Path



# Функция для подготовки данных в пандасе
def dataset_forming(df_train, df_test):
    # Train
    df_train_priz = df_train[['Survived', 'Pclass', 'Age', 'Sex',  'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    df_train_priz['Embarked'] = df_train_priz['Embarked'].fillna('S')
    for col in ['Sex', 'Embarked']:
        df_train_priz[col] = df_train_priz[col].astype('category')
    X_train = df_train_priz.drop('Survived', axis=1)
    # Второй датасет с OneHotCoding
    X_ohe_train = pd.get_dummies(X_train, dtype=float, drop_first=True)
    y_train = df_train_priz['Survived']
    
    fare_median = X_train['Fare'].median()                 
    
            
    # Test    
    df_test_priz = df_test[['Pclass', 'Age', 'Sex',  'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    df_test_priz['Fare'] = df_test_priz['Fare'].fillna(fare_median)
    df_test_priz['Embarked'] = df_test_priz['Embarked'].fillna('S')
    for col in ['Sex', 'Embarked']:
        df_test_priz[col] = df_test_priz[col].astype('category')

    X_test = df_test_priz.copy()
    # Второй датасет с OneHotCoding
    X_ohe_test = pd.get_dummies(X_test, dtype=float, drop_first=True)       
    

    return X_train, X_ohe_train, y_train, X_test, X_ohe_test

# Перебор параметров 
def parametrs(params):
    param = params.copy()
    for key, value in param.items():   
        if type(value) != list:
            param[key] = [value]

    return list(ParameterGrid(param))

# Подготовка данных для валидации
def preparation_train(X_cur, y, train_index, test_index, use_scaler):
    X_train = X_cur.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test  = X_cur.iloc[test_index]
    y_test  = y.iloc[test_index]
            
    age_median = X_train['Age'].median()                 
    X_train['Age'] = X_train['Age'].fillna(age_median)
    X_test['Age'] = X_test['Age'].fillna(age_median)
    
    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


# Подготовка данных для теста
def preparation_inference(X_fit, X_pred, use_scaler):
    X_fit = X_fit.copy()
    X_pred = X_pred.copy()
    
    age_median = X_fit['Age'].median()
    X_fit['Age']  = X_fit['Age'].fillna(age_median)
    X_pred['Age'] = X_pred['Age'].fillna(age_median)
    
    if use_scaler:
        scaler = StandardScaler().fit(X_fit)
        X_fit  = scaler.transform(X_fit)
        X_pred = scaler.transform(X_pred)
        
    return X_fit, X_pred
        
    

# Функции для выгрузки в эксель (все что ниже - писала нейросеть)
# Находим последний путь файла
def next_result_path(folder, exp_name='', prefix='ml_resuts', ext='.xlsx'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)           # создаст папку results, если её нет

    numbers = []
    for f in folder.glob(f'{prefix}*{ext}'):
        m = re.fullmatch(rf'{re.escape(prefix)}(\d+)(?:_.*)?{re.escape(ext)}', f.name)
        if m:
            numbers.append(int(m.group(1)))
    n = max(numbers, default=0) + 1

    name = (exp_name or '').strip()                     # название из конфига
    if name == '':
        name = input('Название эксперимента: ').strip() # пусто -> спрашиваем
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[\\/:*?"<>|]', '', name)

    suffix = f'_{name}' if name else ''
    return folder / f'{prefix}{n}{suffix}{ext}'

# Сервисные колонки 
SERVICE = ['name_model', 'col', 'use_scaler', 'random_state', 'verbose',
           'allow_writing_files', 'cat_features', 'enable_categorical', 'mode', 'batch_size']

def format_results(df, metrics):
    # Порядок колонок: модель, метрики, потом гиперпараметры
    param_cols = [c for c in df.columns if c not in metrics + SERVICE]
    first = ['name_model'] + (['col'] if 'col' in df.columns else [])
    return df[first + metrics + param_cols]

# Выгрузка валидации
def save_validation(results, dl_curves, folder, exp_name=''):
    metrics = ['val_accuracy', 'val_std', 'train_accuracy', 'train_std', 'gap']

    df = pd.DataFrame(results)
    df['gap'] = df['train_accuracy'] - df['val_accuracy']
    df = (format_results(df, metrics)
          .sort_values('val_accuracy', ascending=False)
          .reset_index(drop=True))

    save_path = next_result_path(folder, exp_name)          # ml_resutsN_имя.xlsx
    df.to_excel(save_path, index=False)

    if len(dl_curves) > 0:                                  # кривые только если был DL
        pd.DataFrame(dl_curves).to_csv(
            save_path.with_name(save_path.stem + '_curves.csv'), index=False)
    return save_path

# Выгрузка инференса
def save_inference(submissions, results_test, folder, exp_name=''):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    numbers = []
    for p in folder.glob('inference*'):
        m = re.match(r'inference(\d+)', p.name)
        if p.is_dir() and m:
            numbers.append(int(m.group(1)))
    n = max(numbers, default=0) + 1

    name = re.sub(r'\s+', '_', (exp_name or '').strip())
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    run_dir = folder / (f'inference{n}_{name}' if name else f'inference{n}')
    run_dir.mkdir()

    # Ответы: по файлу на модель (формат Kaggle) + все модели в одной таблице
    for col in submissions.columns.drop('PassengerId'):
        (submissions[['PassengerId', col]]
            .rename(columns={col: 'Survived'})
            .to_csv(run_dir / f'submission_{col}.csv', index=False))
    submissions.to_csv(run_dir / 'all_predictions.csv', index=False)

    # Метрики и параметры: col связывает строку с файлом submission_<col>.csv
    df = pd.DataFrame(results_test)
    df['kaggle_score'] = None                               # впишешь вручную после отправки
    df = format_results(df, ['train_accuracy', 'kaggle_score'])
    df.to_excel(run_dir / 'inference_results.xlsx', index=False)
    return run_dir

