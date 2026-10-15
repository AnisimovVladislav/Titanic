random_state = 42

RandomForestParam = {
    'name_model' : 'RandomForest'
    , 'n_estimators' : 100
    , 'max_depth' : [x for x in range(8, 10, 1)]
    , 'min_samples_leaf' : [x for x in range(1, 4, 1)]
}

LogRegParam = {
    'name_model' : 'LogReg'
}

KNNParam = {
    'name_model' : 'KNN'
    , 'n_neighbors' : [x for x in range(1, 11, 1)]
}

DecisionTreeParam = {
    'name_model' : 'DecisionTree'
    , 'max_depth' : [x for x in range(8, 11, 1)]
    , 'min_samples_leaf' : [x for x in range(1, 4, 1)]
}

path = {
      'train_path'              : r'C:\Users\Vsani\ML\Titanic\data\train.csv'
    , 'test_path'               : r'C:\Users\Vsani\ML\Titanic\data\test.csv'
    , 'gender_submission_path'  : r'C:\Users\Vsani\ML\Titanic\data\gender_submission.csv'
}