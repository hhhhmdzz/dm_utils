import os
import json

from ..utils.base import load_json
from .param_base import param_base

root = os.path.dirname(__file__)

_lgb_default_params = {
    # # bcls
    # 'objective': 'binary',
    # 'metric': 'auc,binary_logloss,binary_error',
    # # mcls
    # 'objective': 'multiclass',
    # 'num_class': num_classes,
    # 'metric': 'multi_logloss,multi_error',
    # # reg
    # 'objective': 'regression',
    # 'metric': 'mae,mse,rmse',

    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_boost_round': 100,  # n_estimators
    'min_split_gain': 0,
    'min_child_samples': 20,  # min_data_in_leaf
    'min_child_weight': 1e-3,  # min_sum_hessian_in_leaf

    'max_depth': -1,
    'num_leaves': 63,
    'bagging_fraction': 0.8,
    'bagging_fraction_seed': param_base['seed'],
    'feature_fraction': 0.8,
    'feature_fraction_seed': param_base['seed'],
    'reg_lambda': 5,  # lambda_l2
    'reg_alpha': 2,  # lambda_l1

    'num_threads': -1,
    'verbose': -1,
}


class LGBModelParams():
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

        self._default_params = _lgb_default_params.copy()

        self._bcls_params = _lgb_default_params.copy()
        self._bcls_params.update({
            'objective': 'binary',
            'metric': 'auc,binary_logloss,binary_error',
        })

        self._mcls_params = _lgb_default_params.copy()
        self._mcls_params.update({
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss,multi_error',
        })

        self._reg_params = _lgb_default_params.copy()
        self._reg_params.update({
            'objective': 'regression',
            'metric': 'mae,mse,rmse',
        })

    def get_params(self, task) -> list:
        if task == 'reg':
            return load_json(os.path.join(root, 'params/lgb/reg.json'))
        elif task == 'cls':
            assert self.num_classes is not None, 'num_classes is not set'
            if self.num_classes == 2:
                return load_json(os.path.join(root, 'params/lgb/bcls.json'))
            return load_json(os.path.join(root, 'params/lgb/mcls.json'))
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
        assert self.num_classes is not None, 'num_classes is not set'
        self._mcls_params.update({'num_class': self.num_classes})
        return self._mcls_params

    @property
    def cls_params(self):
        assert self.num_classes is not None, 'num_classes is not set'
        return self.bcls_params if self.num_classes == 2 else  self.mcls_params

    @property
    def reg_params(self):
        return self._reg_params
