import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet import tab_model as tabnet

_str2skl_cls_model = {
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(),
    'xgb': xgb.XGBClassifier(),
    'xgboost': xgb.XGBClassifier(),
    'lgb': lgb.LGBMClassifier(),
    'lightgbm': lgb.LGBMClassifier(),
    'cb': cb.CatBoostClassifier(),
    'catboost': cb.CatBoostClassifier(),
    'ngb': ngb.NGBClassifier(),
    'ngboost': ngb.NGBClassifier(),
    'tabnet': tabnet.TabNetClassifier(),
}

_str2ori_cls_model = {
    'xgb': xgb.Booster,
    'xgboost': xgb.Booster,
    'lgb': lgb.Booster,
    'lightgbm': lgb.Booster,
    'cb': cb.CatBoost,
    'catboost': cb.CatBoost,
    'ngb': ngb.NGBoost,
    'ngboost': ngb.NGBoost,
}


def get_cls_model_from_str(model_str, sklearn_api):
    if sklearn_api:
        assert model_str in _str2skl_cls_model, f"model_str {model_str} is not supported"
        model = _str2skl_cls_model[model_str]
    else:
        if model_str not in _str2ori_cls_model:
            if model_str not in _str2skl_cls_model:
                raise ValueError(f"model_str {model_str} is not supported")
            else:
                model_str = _str2skl_cls_model[model_str]
        else:
            model = _str2ori_cls_model[model_str]

    return model


def get_cls_data_structure(x, y, model_mode):
    mode1, mode2 = model_mode
    if mode1 == 'sklearn':
        if mode2 == 'tabnet':
            x = x.values
            if y is not None:
                y = y.values
                y = y.reshape(-1)
        elif mode2 == 'ngboost' and len(y.shape) == 2 and y.shape[1] == 1:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            elif isinstance(y, np.ndarray):
                y = y[:, 0]
        data = (x, ) if y is None else (x, y)
    elif mode1 == mode2 == 'xgboost':
        data = xgb.DMatrix(x, label=y)
    elif mode1 == mode2 == 'lightgbm':
        data = (x, ) if y is None else lgb.Dataset(x, label=y)
    elif mode1 == mode2 == 'catboost':
        data = cb.Pool(x, label=y)

    return data


if __name__ == '__main__':
    pass
