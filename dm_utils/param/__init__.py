from .param_base import param_base
from .param_xgb import XGBModelParams
from .param_lgb import LGBModelParams
from .param_cb import CBModelParams


def get_model_params(task, mode2, num_classes=None):
    if mode2 == 'xgboost':
        model_paras = XGBModelParams(num_classes)
    elif mode2 == 'lightgbm':
        model_paras = LGBModelParams(num_classes)
    elif mode2 == 'catboost':
        model_paras = CBModelParams(num_classes)
    else:
        raise ValueError(f'Unknown model: {mode2}')

    if task == 'reg':
        return model_paras.reg_params
    elif task == 'cls':
        return model_paras.cls_params
    else:
        raise ValueError(f'Unknown task: {task}')


def get_gpu_params(mode2, num_classes=None):
    params = dict()
    if mode2 in {'xgb', 'xgboost'}:
        params['tree_method'] = 'hist'  # gpu_hist
        params['device'] = 'cuda'
    elif mode2 in {'lgb', 'lightgbm'}:
        # params['gpu_platform_id'] = 0
        # params['gpu_device_id'] = 0
        params['device'] = 'gpu'
    elif mode2 in {'cb', 'catboost'}:
        if num_classes is not None and num_classes > 2:  # multi-class with gpu seem not supported in catboost
            pass
        else:  # regression or binary classification
            params['task_type'] = 'GPU'
            # params['devices'] = '0'
            # params['gpu_ram_part'] = 0.95
    else:
        pass
        # raise ValueError(f'Unknown model: {mode2}')
    
    return params
