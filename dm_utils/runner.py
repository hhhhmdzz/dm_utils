import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb

from .evaluate.score import logits2prob
from .utils.base import get_model_mode, get_log_level
from .task import get_model_from_str, get_data_structure
from .param import get_model_params, get_gpu_params


def train(task, model, x, y, x_evals, y_evals, params=None, epochs=1000, eval_rounds=100, early_stop_rounds=1000, log_level=0, use_gpu4tree=False):
    """

    Args:
        task
        model
        x
        y
        x_evals
        y_evals
        params
        epochs
        eval_rounds
        early_stop_rounds
        log_level: [-1, 0, 1, 2, 3]
    
    Returns:
        model
    """
    mode1, mode2 = get_model_mode(model)
    data = get_data_structure(x, y, (mode1, mode2), task)
    data_evals = get_data_structure(x_evals, y_evals, (mode1, mode2), task)

    params = dict() if params is None else params
    if mode1 != 'sklearn':
        _num_classes = len(np.unique(y)) if task == 'cls' else None
        _model_params = get_model_params(task, mode2, num_classes=_num_classes)
        for k, v in _model_params.items():
            if k not in params:
                params[k] = v
        if use_gpu4tree:
            _gpu_params = get_gpu_params(mode2, _num_classes)
            for k, v in _gpu_params.items():
                if k not in params:
                    params[k] = v

    if mode1 == 'sklearn':
        if mode2 == 'xgboost':
            model.fit(*data, eval_set=[data_evals], verbose=eval_rounds, **params)
        elif mode2 == 'lightgbm':
            callbacks = [lgb.log_evaluation(period=eval_rounds), lgb.early_stopping(stopping_rounds=early_stop_rounds)]
            model.fit(*data, eval_set=data_evals, eval_names='valid', callbacks=callbacks, **params)
        elif mode2 == 'catboost':
            model.fit(*data, eval_set=data_evals, verbose_eval=eval_rounds, early_stopping_rounds=early_stop_rounds, **params)
        elif mode2 == 'ngboost':
            model.fit(*data, *data_evals, **params)
        elif mode2 == 'tabnet':
            model.fit(*data, eval_set=[data_evals], max_epochs=epochs, patience=early_stop_rounds, **params)
        else:
            model.fit(*data, **params)
    elif mode1 == 'xgboost':
        params['verbosity'] = get_log_level(mode2, log_level)
        model = xgb.train(params=params, dtrain=data, evals=[(data_evals, 'valid'), ], num_boost_round=epochs, verbose_eval=eval_rounds, early_stopping_rounds=early_stop_rounds)
    elif mode1 == 'lightgbm':
        params['verbosity'] = get_log_level(mode2, log_level)
        params['num_boost_round'] = epochs
        callbacks = [lgb.log_evaluation(period=eval_rounds), lgb.early_stopping(stopping_rounds=early_stop_rounds)]
        model = lgb.train(params=params, train_set=data, valid_sets=data_evals, callbacks=callbacks)
    elif mode1 == 'catboost':
        params['logging_level'] = get_log_level(mode2, log_level)
        model = cb.train(params=params, dtrain=data, eval_set=data_evals, iterations=epochs, verbose_eval=eval_rounds, early_stopping_rounds=early_stop_rounds)
    
    return model


def predict(task, model, x):
    mode1, mode2 = get_model_mode(model)

    if task == 'cls':
        if mode1 == 'sklearn':
            prediction = model.predict_proba(x.values)
        elif mode1 == 'lightgbm':
            prediction = model.predict(x)
        elif mode1 in {'xgboost', 'catboost'}:
            data = get_data_structure(x, None, (mode1, mode2), task)
            prediction = model.predict(data)
            if mode1 == 'catboost':
                prediction = logits2prob(prediction)

        if len(prediction.shape) == 2 and prediction.shape[1] == 2:
            prediction = prediction[:, 1]
    elif task == 'reg':
        if mode1 in {'sklearn', 'lightgbm'}:
            prediction = model.predict(x.values)
        elif mode1 in {'xgboost', 'catboost'}:
            data = get_data_structure(x, None, (mode1, mode2), task)
            prediction = model.predict(data)

        if mode2 == 'tabnet':
            prediction = prediction.squeeze()
    else:
        raise ValueError(f'Invalid task: {task}')

    return prediction


if __name__ == '__main__':
    pass
