from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from pathlib import Path

import DeepLearning as DL

random_state = 42
exp_name = 'Тест baseline'
mode = 'inference' ## Выбрать validation / inference
n_splits = 5

models = {
    'RandomForest': RandomForestClassifier
    , 'DecisionTree': DecisionTreeClassifier
    , 'LogReg': LogisticRegression
    , 'KNN': KNeighborsClassifier
    , 'CatBoost' : CatBoostClassifier
    , 'LGBM' : LGBMClassifier
    , 'XGBoost' : XGBClassifier
    , 'nn_model' : DL.nn_model
}

rename_params = {
    'iterations'        : 'n_estimators'       # CatBoost
    , 'depth'           : 'max_depth'          # CatBoost
    , 'l2_leaf_reg'     : 'reg_lambda'         # CatBoost
    , 'min_child_samples': 'min_samples_leaf'  # LGBM: тот же смысл, минимум объектов в листе
}

RandomForestParam = {
    'name_model' : 'RandomForest'
    , 'random_state' : random_state
    , 'use_scaler' : False
    , 'use_category' : False
    , 'n_estimators' : 200
    , 'max_depth' : 10
    , 'min_samples_leaf' : 2
    , 'max_features' : 0.5
}

DecisionTreeParam = {
    'name_model' : 'DecisionTree'
    , 'random_state' : random_state
    , 'use_scaler' : False
    , 'use_category' : False
    , 'max_depth' : 8
    , 'min_samples_leaf' : 5
    , 'criterion' : 'gini'
}

LogRegParam = {
    'name_model' : 'LogReg'
    , 'random_state' : random_state
    , 'use_scaler' : True
    , 'use_category' : False
    , 'C' : [0.003, 0.01, 0.03]
    , 'max_iter' : 1000
}

KNNParam = {
    'name_model' : 'KNN'
    , 'use_scaler' : True
    , 'use_category' : False
    , 'n_neighbors' : [9, 11]
    , 'weights' : 'uniform'
    , 'p' : 1
}

CatBoostParam = {
    'name_model' : 'CatBoost'
    , 'random_state' : random_state
    , 'use_scaler' : False
    , 'use_category' : True
    , 'cat_features' : ('Sex', 'Embarked')
    , 'verbose' : 0
    , 'allow_writing_files' : False
    , 'iterations' : 300
    , 'learning_rate' : 0.1
    , 'depth' : 6
    , 'l2_leaf_reg' : 3
}

LGBMParam = {
    'name_model' : 'LGBM'
    , 'random_state' : random_state
    , 'use_scaler' : False
    , 'use_category' : True
    , 'verbose' : -1
    , 'n_estimators' : 100
    , 'learning_rate' : 0.1
    , 'num_leaves' : 8
    , 'min_child_samples' : 20
}

XGBoostParam = {
    'name_model' : 'XGBoost'
    , 'random_state' : random_state
    , 'use_scaler' : False
    , 'use_category' : True
    , 'enable_categorical' : True
    , 'n_estimators' : 500
    , 'learning_rate' : 0.03
    , 'max_depth' : 4
    , 'min_child_weight' : 5
    , 'subsample' : 0.8
}

DLParam = {
    'name_model' : 'nn_model'
    , 'random_state' : random_state
    , 'use_scaler' : True
    , 'use_category' : False
    , 'EPOCHS' : 100
    , 'lr' : 1e-3
    , 'weight_decay' : [1e-1]
    , 'mode' : 'min'
    , 'factor' : [0.5]
    , 'patience' : 5
    , 'threshold' : 1e-2
    , 'min_lr' : 1e-7
    , 'batch_size' : 64
}

BASE = Path(__file__).parent
path = {
    'train_path': BASE / 'data' / 'train.csv',
    'test_path':  BASE / 'data' / 'test.csv',
    'file_path':  BASE / 'results',
}