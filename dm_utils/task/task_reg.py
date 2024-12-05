import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet import tab_model as tabnet

from dm_utils.utils import cprint as uu_print
from dm_utils.param.set_params import set_params

_str2skl_reg_model = {
    'lr': LinearRegression(),
    'svm': SVR(),
    'knn': KNeighborsRegressor(),
    'dt': DecisionTreeRegressor(),
    'et': ExtraTreeRegressor(),
    'rf': RandomForestRegressor(),
    'ets': ExtraTreesRegressor(),
    'gb': GradientBoostingRegressor(),
    'hgb': HistGradientBoostingRegressor(),
    'xgb': xgb.XGBRegressor(),
    'xgboost': xgb.XGBRegressor(),
    'lgb': lgb.LGBMRegressor(),
    'lightgbm': lgb.LGBMRegressor(),
    'cb': cb.CatBoostRegressor(),
    'catboost': cb.CatBoostRegressor(),
    'ngb': ngb.NGBRegressor,
    'ngboost': ngb.NGBRegressor,
    'tabnet': tabnet.TabNetRegressor(),
}
for k in _str2skl_reg_model.keys():
    if k not in {'ngb', 'ngboost'}:
        set_params(_str2skl_reg_model[k])

_str2ori_reg_model = {
    'xgb': xgb.Booster,
    'xgboost': xgb.Booster,
    'lgb': lgb.Booster,
    'lightgbm': lgb.Booster,
    'cb': cb.CatBoost,
    'catboost': cb.CatBoost,
    # 'ngb': ngb.NGBoost,
    # 'ngboost': ngb.NGBoost,
}


def get_reg_model_from_str(model_str, sklearn_api):
    if sklearn_api:
        assert model_str in _str2skl_reg_model, f"model_str '{model_str}' is not supported"
        model = _str2skl_reg_model[model_str]
    else:
        if model_str not in _str2ori_reg_model:
            if model_str not in _str2skl_reg_model:
                raise ValueError(f"model_str '{model_str}' is not supported")
            else:
                uu_print.conflict(f"model_str '{model_str}' is not supported when sklearn_api=False, automatically use sklearn api.")
                model = _str2skl_reg_model[model_str]
        else:
            model = _str2ori_reg_model[model_str]

    return deepcopy(model)


def get_reg_data_structure(x, y, model_mode, weight=None):
    mode1, mode2 = model_mode
    if mode1 == 'sklearn':
        if mode2 == 'tabnet':
            x = x.values
            if y is not None:
                y = y.values
                y = y.reshape(-1, 1)
        elif mode2 == 'ngboost' and len(y.shape) == 2 and y.shape[1] == 1:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            elif isinstance(y, np.ndarray):
                y = y[:, 0]
        data = (x, ) if y is None else (x, y)
    elif mode1 == mode2 == 'xgboost':
        data = xgb.DMatrix(x, label=y, weight=weight)
    elif mode1 == mode2 == 'lightgbm':
        data = (x, ) if y is None else lgb.Dataset(x, label=y, weight=weight)
    elif mode1 == mode2 == 'catboost':
        data = cb.Pool(x, label=y, weight=weight)

    return data


if __name__ == '__main__':
    pass
