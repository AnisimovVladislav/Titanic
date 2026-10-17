import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

import config as cnf 


# Перебор параметров 
def parametrs(params):
    param = params.copy()
    for key, value in param.items():   
        if type(value) != list:
            param[key] = [value]

    return list(ParameterGrid(param))

