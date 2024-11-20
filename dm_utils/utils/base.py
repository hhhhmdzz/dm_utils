import os
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet import tab_model as tabnet


def is_sklearn_mode(model):
    cb_sklearn_model = (cb.CatBoostClassifier, cb.CatBoostRegressor, cb.CatBoostRanker)
    return isinstance(model, (BaseEstimator, *cb_sklearn_model))


def is_xgboost_mode(model):
    return isinstance(model, (xgb.Booster, xgb.XGBModel)) or model == xgb.Booster


def is_lightgbm_mode(model):
    return isinstance(model, (lgb.Booster, lgb.LGBMModel)) or model == lgb.Booster


def is_catboost_mode(model):
    return isinstance(model, cb.CatBoost) or model == cb.CatBoost


def is_ngboost_mode(model):
    return isinstance(model, ngb.NGBoost)


def is_tabnet_mode(model):
    return isinstance(model, TabModel)


def get_model_mode(model):
    if is_sklearn_mode(model):
        mode1 = 'sklearn'
        if is_xgboost_mode(model):
            mode2 = 'xgboost'
        elif is_lightgbm_mode(model):
            mode2 = 'lightgbm'
        elif is_catboost_mode(model):
            mode2 = 'catboost'
        elif is_ngboost_mode(model):
            mode2 = 'ngboost'
        elif is_tabnet_mode(model):
            mode2 = 'tabnet'
        else:
            mode2 = 'sklearn'
    else:
        if is_xgboost_mode(model):
            mode1 = mode2 = 'xgboost'
        elif is_lightgbm_mode(model):
            mode1 = mode2 = 'lightgbm'
        elif is_catboost_mode(model):
            mode1 = mode2 = 'catboost'
        else:
            raise ValueError(f"model {model} is not supported")

    return mode1, mode2


def get_model_name(model):
    mode1, mode2 = get_model_mode(model)
    if mode1 == 'sklearn':
        return model.__class__.__name__
    else:
        if mode2 == 'xgboost':
            return 'XGBoost'
        elif mode2 == 'lightgbm':
            return 'LightGBM'
        elif mode2 == 'catboost':
            return 'CatBoost'
        else:
            raise ValueError(f"model {model} is not supported")


_xgb_log_level = {  # verbosity
    -1: 0,
    0: 0,  # silent
    1: 1,  # warning
    2: 2,  # info
    3: 3,  # debug
}
_lgb_log_level = {  # verbosity
    -1: -1,
    0: -1,  # silent
    1: 0,  # warning
    2: 1,  # info
    3: 2,  # debug
}
_cb_log_level = {  # logging_level
    -1: 'Silent',  # 不输出信息
    0: 'Verbose',  #  输出评估指标、已训练时间、剩余时间等
    1: 'Info',  # 输出额外信息、树的棵树
    2: 'Debug',  # debug信息
    3: 'Debug',
}


def get_log_level(mode2, log_level):
    if mode2 in {'xgb', 'xgboost'}:
        return _xgb_log_level[log_level]
    elif mode2 in {'lgb', 'lightgbm'}:
        return _lgb_log_level[log_level]
    elif mode2 in {'cb', 'catboost'}:
        return _cb_log_level[log_level]
    else:
        raise ValueError(f"model {mode2} is not supported")


def save_json(data, file_path):
    json.dump(data, open(file_path, 'w'), indent=4)


def load_json(file_path):
    return json.load(open(file_path, 'r'))


if __name__ == '__main__':
    pass