from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

random_state = 42

models = {
    'RandomForest': RandomForestClassifier
    , 'DecisionTree': DecisionTreeClassifier
    , 'LogReg': LogisticRegression
    , 'KNN': KNeighborsClassifier
}

RandomForestParam = {
    'name_model' : 'RandomForest'
    , 'n_estimators' : 100
    , 'max_depth' : [x for x in range(8, 10, 1)]
    , 'min_samples_leaf' : [x for x in range(1, 4, 1)]
    , 'scaler' : False
    , 'random_state' : random_state
}

DecisionTreeParam = {
    'name_model' : 'DecisionTree'
    , 'max_depth' : [x for x in range(8, 11, 1)]
    , 'min_samples_leaf' : [x for x in range(1, 4, 1)]
    , 'scaler' : False
    , 'random_state' : random_state
}

LogRegParam = {
    'name_model' : 'LogReg'
    , 'scaler' : True
    , 'random_state' : random_state
}

KNNParam = {
    'name_model' : 'KNN'
    , 'n_neighbors' : [x for x in range(1, 11, 1)]
    , 'scaler' : True
}


path = {
      'train_path'              : r'C:\Users\Vsani\ML\Titanic\data\train.csv'
    , 'test_path'               : r'C:\Users\Vsani\ML\Titanic\data\test.csv'
    , 'gender_submission_path'  : r'C:\Users\Vsani\ML\Titanic\data\gender_submission.csv'
}