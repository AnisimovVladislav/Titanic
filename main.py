import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cnf 
import utils as utl
import DeepLearning as DL


# Перебор гиперпараметров
def build_combo_param():
    combo_param = utl.parametrs(cnf.RandomForestParam)
    combo_param += utl.parametrs(cnf.DLParam)
    combo_param += utl.parametrs(cnf.LogRegParam)
    combo_param += utl.parametrs(cnf.KNNParam)
    combo_param += utl.parametrs(cnf.DecisionTreeParam)
    combo_param += utl.parametrs(cnf.CatBoostParam)
    combo_param += utl.parametrs(cnf.LGBMParam)
    combo_param += utl.parametrs(cnf.XGBoostParam)
    return combo_param

# ------------------ Цикл валидации ------------------
def run_validation(X, X_ohe, y, combo_param):
    results = []
    dl_curves = [] 
    SK_Fold = StratifiedKFold(n_splits=cnf.n_splits, shuffle=True, random_state=cnf.random_state)
    
    # Перебор всех гиперпараметров
    for k, combo in enumerate(combo_param):

        params = combo.copy()
        name_model = params.pop('name_model')
        use_scaler = params.pop('use_scaler')
        use_category = params.pop('use_category')

        # Флаг на категориальные признаки - если False, делаем OneHotCoding
        X_cur = X if use_category else X_ohe

        array_train_accuracy = []
        array_val_accuracy = []

        # Запускаем DL
        if name_model == 'nn_model':
            
            for i, (train_index, val_index) in enumerate(SK_Fold.split(X, y)):

                # Объявляем модель
                loss_model = nn.CrossEntropyLoss()
                torch.manual_seed(cnf.random_state)

                model = DL.nn_model(X_cur.shape[1])
                opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                opt
                                                                , mode=params['mode']
                                                                , factor=params['factor']
                                                                , patience=params['patience']
                                                                , threshold=params['threshold']
                                                                , min_lr=params['min_lr']
                                                                )

                # Подготовка данных для обучения
                X_train, y_train, X_val, y_val = utl.preparation_train(X_cur, y, train_index, val_index, use_scaler)

                tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
                tensor_X_val  = torch.tensor(X_val,  dtype=torch.float32)
                tensor_y_train = torch.tensor(y_train.values, dtype=torch.long)
                tensor_y_val  = torch.tensor(y_val.values,  dtype=torch.long)

                dataset = TensorDataset(tensor_X_train, tensor_y_train)
                dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

                # Цикл по эпохам
                train_loop = tqdm(range(params['EPOCHS']))
                for epoch in train_loop:

                    model.train()

                    for _, val in enumerate(dataloader):
                    
                        # Прямой проход
                        pred = model(val[0])
                        loss = loss_model(pred, val[1])

                        # Обратный проход
                        opt.zero_grad()
                        loss.backward()

                        # Шаг оптимизации
                        opt.step()

                    model.eval()

                    # Считаем на валидации
                    with torch.no_grad():
                        # Считаем предсказания
                        pred_train = model(tensor_X_train)
                        pred_val = model(tensor_X_val)

                        # Считаем лосс
                        loss_train = loss_model(pred_train, tensor_y_train)
                        loss_val = loss_model(pred_val, tensor_y_val)

                        # Считаем точность на трейне
                        train_loss = loss_train.item()
                        true_answer = (pred_train.argmax(dim=1) == tensor_y_train).sum().item()
                        train_accuracy = true_answer / len(tensor_X_train)

                        # Считаем точность на тесте
                        val_loss = loss_val.item()
                        true_answer = (pred_val.argmax(dim=1) == tensor_y_val).sum().item()
                        val_accuracy = true_answer / len(tensor_X_val)

                    train_loop.set_description(f"Epoch [{epoch+1}/{params['EPOCHS']}], train_loss={train_loss:2f}, train_accuracy={train_accuracy:2f}, val_loss={val_loss:2f} val_accuracy={val_accuracy:2f} lr={scheduler.get_last_lr()[0]}")

                    scheduler.step(train_loss)


                    dl_curves.append({

                        'name_model': name_model
                        , 'Fold': i
                        , 'epoch': epoch
                        , 'train_accuracy': train_accuracy  
                        , 'train_loss': train_loss
                        , 'val_accuracy': val_accuracy
                        , 'val_loss': val_loss
                        , 'lr': opt.param_groups[0]['lr']
                        , 'combo': f"weight_decay={params['weight_decay']}, factor={params['factor']}"
                    })

                array_train_accuracy.append(train_accuracy)   # значение последней эпохи
                array_val_accuracy.append(val_accuracy)


        # Запускаем линейные модели и деревья
        else:     

            model = cnf.models[name_model](**params)
            
            train_loop = tqdm(enumerate(SK_Fold.split(X, y)))
            for i, (train_index, val_index) in train_loop:
                
                # Подготовка данных для обучения
                X_train, y_train, X_val, y_val = utl.preparation_train(X_cur, y, train_index, val_index, use_scaler)

                model_clone = clone(model)
                model_clone.fit(X_train, y_train)

                # Считаем accuracy
                train_accuracy = model_clone.score(X_train, y_train)
                val_accuracy = model_clone.score(X_val, y_val)

                array_train_accuracy.append(train_accuracy)
                array_val_accuracy.append(val_accuracy)
                                
                train_loop.set_description(f"SK_Fold [{i+1}/{cnf.n_splits}], model {name_model}, train_accuracy={np.mean(array_train_accuracy):2f}, val_accuracy={np.mean(array_val_accuracy):2f}")
        
        # Считаем accuracy для текущих модель-гиперпараметры
        mean_train_accuracy = float(np.mean(array_train_accuracy))
        mean_train_std = float(np.std(array_train_accuracy))
        mean_val_accuracy = float(np.mean(array_val_accuracy))
        mean_val_std = float(np.std(array_val_accuracy))
        
        # Переименовываем параметры в Катбусте
        combo_row = {cnf.rename_params.get(k, k): v for k, v in combo.items()}

        results.append({
            **combo_row
            , 'col': f'{name_model}_{k}' 
            , 'train_accuracy'  :   mean_train_accuracy
            , 'train_std'       :   mean_train_std
            , 'val_accuracy'   :   mean_val_accuracy
            , 'val_std'        :   mean_val_std
        })
    
    # Выгрузка результатов и обучения ДЛ в эксель
    return utl.save_validation(results, dl_curves, cnf.path['file_path'], cnf.exp_name)
    
    




# ------------------ Цикл инференеса ------------------
def run_inference(X, X_ohe, y, X_test, X_ohe_test, df_test, combo_param):
    
    submissions = pd.DataFrame({'PassengerId': df_test['PassengerId']})
    results_test = []

    # Перебор гиперпараметров
    for k, combo in enumerate(combo_param):
        params = combo.copy()
        name_model = params.pop('name_model')
        use_scaler = params.pop('use_scaler')
        use_category = params.pop('use_category')
        
        # Обучение ДЛ
        if name_model == 'nn_model':
            
            # Подготовка данных
            X_fit, X_pred = utl.preparation_inference(
                                X if use_category else X_ohe,
                                X_test if use_category else X_ohe_test,
                                use_scaler)
            
            tensor_X_fit  = torch.tensor(np.asarray(X_fit,  dtype=np.float32))
            tensor_X_pred = torch.tensor(np.asarray(X_pred, dtype=np.float32))
            tensor_y      = torch.tensor(y.values, dtype=torch.long)
            
            
            # Объявляем модель
            loss_model = nn.CrossEntropyLoss()
            torch.manual_seed(cnf.random_state)

            model = DL.nn_model(X_fit.shape[1])
            opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            opt
                                                            , mode=params['mode']
                                                            , factor=params['factor']
                                                            , patience=params['patience']
                                                            , threshold=params['threshold']
                                                            , min_lr=params['min_lr']
                                                            )


            dataset = TensorDataset(tensor_X_fit, tensor_y)
            dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

            train_loop = tqdm(range(params['EPOCHS']))
            for epoch in train_loop:

                model.train()

                for _, val in enumerate(dataloader):
                    
                    # Прямой проход
                    pred = model(val[0])
                    loss = loss_model(pred, val[1])

                    # Обратный проход
                    opt.zero_grad()
                    loss.backward()

                    # Шаг оптимизации
                    opt.step()

                model.eval()

                # Считаем на валидации
                with torch.no_grad():
                    # Считаем предсказания
                    pred_train = model(tensor_X_fit)

                    # Считаем лосс
                    loss_train = loss_model(pred_train, tensor_y)

                    # Считаем точность на трейне
                    train_loss = loss_train.item()
                    true_answer = (pred_train.argmax(dim=1) == tensor_y).sum().item()
                    train_accuracy = true_answer / len(tensor_X_fit)

                train_loop.set_description(f"Epoch [{epoch+1}/{params['EPOCHS']}], train_loss={train_loss:2f}, train_accuracy={train_accuracy:2f}, lr={scheduler.get_last_lr()[0]}")

                scheduler.step(train_loss)


            model.eval()
            # Считаем итоговый train_accuracy и получаем ответы модели на тесте
            with torch.no_grad():
                train_accuracy = (model(tensor_X_fit).argmax(dim=1) == tensor_y).float().mean().item()
                answer = model(tensor_X_pred).argmax(dim=1).numpy()
            
            
            col = f'{name_model}_{k}'
            submissions[col] = answer
            combo_row = {cnf.rename_params.get(kk, kk): v for kk, v in combo.items()}
            results_test.append({
                    **combo_row
                    , 'col': col
                    , 'train_accuracy': train_accuracy
                })           
            
        # Линейные модели и деревья
        else:
            
            # Флаг на категориальные признаки - если False, делаем OneHotCoding
            # Подготовка данных
            X_fit, X_pred = utl.preparation_inference(
                                X if use_category else X_ohe,
                                X_test if use_category else X_ohe_test,
                                use_scaler)

            model = clone(cnf.models[name_model](**params)).fit(X_fit, y)
            train_accuracy = model.score(X_fit, y)

            # Сохраняем данные
            col = f'{name_model}_{k}'
            submissions[col] = model.predict(X_pred)   
            # Переименовываем Катбуст
            combo_row = {cnf.rename_params.get(kk, kk): v for kk, v in combo.items()}

            results_test.append({
                    **combo_row
                    , 'col': col
                    , 'train_accuracy'  :   train_accuracy
                })
            
            print(f"model {name_model}, iter {k+1}/{len(combo_param)}, train_accuracy={train_accuracy:2f}")
                    
            
    # Выгружаем результат и ответы моделей
    return utl.save_inference(submissions, results_test, cnf.path['file_path'], cnf.exp_name)


def main():
    assert cnf.mode in ('validation', 'inference')

    df_train = pd.read_csv(cnf.path['train_path'])
    df_test  = pd.read_csv(cnf.path['test_path'])
    X, X_ohe, y, X_test, X_ohe_test = utl.dataset_forming(df_train, df_test)
    combo_param = build_combo_param()

    if cnf.mode == 'validation':
        out = run_validation(X, X_ohe, y, combo_param)
    else:
        out = run_inference(X, X_ohe, y, X_test, X_ohe_test, df_test, combo_param)

    print(out)


if __name__ == '__main__':
    main()
