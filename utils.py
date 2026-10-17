from sklearn.model_selection import ParameterGrid

import re
from pathlib import Path

import config as cnf 

# Перебор параметров 
def parametrs(params):
    param = params.copy()
    for key, value in param.items():   
        if type(value) != list:
            param[key] = [value]

    return list(ParameterGrid(param))



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

