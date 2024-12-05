import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
import torch
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet import tab_model as tabnet


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # use deterministic convolution algorithm
    # True: easier to reproduce results
    # False(default): faster
    torch.backends.cudnn.deterministic = True
    # auto tune the algorithm
    # True(default): improves performance, not easy to reproduce results
    # False: imporves training speed, easy to reproduce results
    torch.backends.cudnn.benchmark = False
    # enable cudnn
    # True(default):
    # False:
    torch.backends.cudnn.enabled = True
    # faster but less accurate
    # True:
    # False(default after 1.12):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def is_sklearn_mode(model):
    is_skl = is_cb = is_ngb = False
    if type(model) is type:  # class
        is_ngb = issubclass(model, ngb.NGBoost)
    else:  # instance
        is_skl = isinstance(model, BaseEstimator)
        is_cb = isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRegressor, cb.CatBoostRanker))
    return is_skl or is_cb or is_ngb


def is_xgboost_mode(model):
    is_skl = is_ori = False
    if type(model) is type:
        is_ori = issubclass(model, xgb.Booster)
    else:
        is_skl = isinstance(model, (xgb.XGBModel, xgb.Booster))
    return is_skl or is_ori


def is_lightgbm_mode(model):
    is_skl = is_ori = False
    if type(model) is type:
        is_ori = issubclass(model, lgb.Booster)
    else:
        is_skl = isinstance(model, (lgb.LGBMModel, lgb.Booster))
    return is_skl or is_ori


def is_catboost_mode(model):
    is_skl = is_ori = False
    if type(model) is type:
        is_ori = issubclass(model, cb.CatBoost)
    else:
        is_skl = isinstance(model, cb.CatBoost)
    return is_skl or is_ori


def is_ngboost_mode(model):
    if type(model) is type:
        return issubclass(model, ngb.NGBoost)
    else:
        return isinstance(model, ngb.NGBoost)


def is_tabnet_mode(model):
    if type(model) is type:
        return False
    else:
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
        if mode2 == 'ngboost' and type(model) is type:
            if issubclass(model, ngb.NGBClassifier):
                return 'NGBClassifier'
            elif issubclass(model, ngb.NGBRegressor):
                return 'NGBRegressor'
            elif issubclass(model, ngb.NGBSurvival):
                return 'NGBSurvival'
            else:
                raise ValueError(f"model {model} is not supported")
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
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    print('regression:')
    models = [
        DecisionTreeRegressor(),
        xgb.XGBRegressor(),
        lgb.LGBMRegressor(),
        cb.CatBoostRegressor(),
        ngb.NGBRegressor,
        ngb.NGBRegressor(),
        tabnet.TabNetRegressor(),
        xgb.Booster,
        lgb.Booster,
        cb.CatBoost,
    ]
    for model in models:
        print(get_model_mode(model))

    print('classification:')
    models = [
        DecisionTreeClassifier(),
        xgb.XGBClassifier(),
        lgb.LGBMClassifier(),
        cb.CatBoostClassifier(),
        ngb.NGBClassifier,
        ngb.NGBClassifier(),
        tabnet.TabNetClassifier(),
        xgb.Booster,
        lgb.Booster,
        cb.CatBoost,
    ]
    for model in models:
        print(get_model_mode(model))
