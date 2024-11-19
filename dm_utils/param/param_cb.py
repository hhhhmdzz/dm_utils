import os
import json

from ..utils.base import load_json
from .param_base import param_base

root = os.path.dirname(__file__)

_cb_default_params = {
    # # bcls
    # 'loss_function': 'Logloss',
    # 'custom_metric': ['F1', 'Accuracy', 'AUC'],
    # 'eval_metric': 'AUC',
    # # mcls
    # 'loss_function': 'MultiClass',
    # # 'num_class': num_classes,
    # 'custom_metric': ['MultiClass', 'Accuracy'],
    # 'eval_metric': 'MultiClass',
    # # reg
    # 'loss_function': 'RMSE',
    # 'custom_metric': ['RMSE', 'MAE'],
    # 'eval_metric': 'MAE',

    # 'boosting_type': 'Auto',  # Auto Ordered Plain
    'learning_rate': 0.03,
    'iterations': 1000,  # num_boost_round
    # min_split_gain
    'min_data_in_leaf': 1,  # min_child_samples
    # min_child_weight
    'rsm': 1,
    'leaf_estimation_method': 'Gradient',  # Newton Gradient Exact
    'one_hot_max_size': 2,
    'fold_len_multiplier': 2,

    'depth': 6,
    'max_leaves': 31,  # num_leaves
    'bagging_temperature': 1,
    'random_seed': param_base['seed'],
    'random_strength': 1,
    'l2_leaf_reg': 3,  # reg_lambda
    # reg_alpha

    'border_count': 128,

    'thread_count': -1,
    'use_best_model': True,
    'logging_level': 'Verbose',
}


class CBModelParams():
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

        self._default_params = _cb_default_params.copy()

        self._bcls_params = _cb_default_params.copy()
        self._bcls_params.update({
            'loss_function': 'Logloss',
            'custom_metric': ['F1', 'Accuracy', 'AUC'],
            'eval_metric': 'AUC',
        })

        self._mcls_params = _cb_default_params.copy()
        self._mcls_params.update({
            'loss_function': 'MultiClass',
            'custom_metric': ['MultiClass', 'Accuracy'],
            'eval_metric': 'MultiClass',
        })

        self._reg_params = _cb_default_params.copy()
        self._reg_params.update({
            'loss_function': 'RMSE',
            'custom_metric': ['RMSE', 'MAE'],
            'eval_metric': 'MAE',
        })

    def get_params(self, task) -> list:
        if task == 'reg':
            return load_json(os.path.join(root, 'params/cb/reg.json'))
        elif task == 'cls':
            assert self.num_classes is not None, 'num_classes is not set'
            if self.num_classes == 2:
                return load_json(os.path.join(root, 'params/cb/bcls.json'))
            return load_json(os.path.join(root, 'params/cb/mcls.json'))
        else:
            raise ValueError(f'Invalid task: {task}')

    @property
    def default_params(self):
        return self._default_params

    @property
    def bcls_params(self):
        return self._bcls_params

    @property
    def mcls_params(self):
        return self._mcls_params

    @property
    def cls_params(self):
        assert self.num_classes is not None, 'num_classes is not set'
        return self.bcls_params if self.num_classes == 2 else  self.mcls_params

    @property
    def reg_params(self):
        return self._reg_params
