import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet import tab_model as tabnet

from .task_cls import get_cls_model_from_str, get_cls_data_structure
from .task_reg import get_reg_model_from_str, get_reg_data_structure


def get_model_from_str(task, model_str, sklearn_api):
    if task == 'cls':
        model = get_cls_model_from_str(model_str, sklearn_api)
    elif task == 'reg':
        model = get_reg_model_from_str(model_str, sklearn_api)

    return model


def get_data_structure(x, y, model_mode, task):
    task_set = {'cls', 'reg'}
    assert task in task_set, f"task must be in {task_set}, but got {task}"
    
    mode1, mode2 = model_mode
    mode1_set = {'sklearn', 'xgboost', 'lightgbm', 'catboost'}
    assert mode1 in mode1_set, f"model_mode must be in {mode1_set}, but got {mode1}"
    mode2_set = {'sklearn', 'ngboost', 'tabnet', 'xgboost', 'lightgbm', 'catboost'}
    assert mode2 in mode2_set, f"model_mode must be in {mode2_set}, but got {mode2}"

    if task == 'cls':
        data = get_cls_data_structure(x, y, model_mode)
    elif task == 'reg':
        data = get_reg_data_structure(x, y, model_mode)

    return data


if __name__ == '__main__':
    pass
