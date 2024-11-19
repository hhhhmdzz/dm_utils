import os
import json

from ..utils.base import load_json
from .param_base import param_base

root = os.path.dirname(__file__)

_xgb_default_params = {
    # # bcls
    # 'objective': 'binary:logistic',
    # 'metric': 'auc',
    # # mcls
    # 'objective': 'multi:softprob',
    # 'num_class': num_classes,
    # 'metric': 'auc',
    # # reg
    # 'objective': 'reg:squarederror',
    # 'eval_metric': 'mae',

    'booster': 'gbtree',
    'eta': 0.03,  # lr
    # epoch
    'gamma': 0.1,  # min_split_gain
    # min_data_in_leaf
    'min_child_weight': 3,

    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'colsample_bylevel': 1.0,
    'colsample_bynode': 1.0,
    'lambda': 10,
    'alpha': 0,

    'missing': -999.0,
    'seed': param_base['seed'],
    'nthread': -1,
    'verbosity': 0,
}


class XGBModelParams():
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

        self._default_params = _xgb_default_params.copy()

        self._bcls_params = _xgb_default_params.copy()
        self._bcls_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'metric': 'auc'
        })

        self._mcls_params = _xgb_default_params.copy()
        self._mcls_params.update({
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'eval_metric': 'auc',
            'metric': 'auc'
        })

        self._reg_params = _xgb_default_params.copy()
        self._reg_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
        })

    def get_params(self, task) -> list:
        if task == 'reg':
            return load_json(os.path.join(root, 'params/xgb/reg.json'))
        elif task == 'cls':
            assert self.num_classes is not None, 'num_classes is not set'
            if self.num_classes == 2:
                return load_json(os.path.join(root, 'params/xgb/bcls.json'))
            return load_json(os.path.join(root, 'params/xgb/mcls.json'))
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
