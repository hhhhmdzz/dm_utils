from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import BaseEnsemble
from ngboost.distns import k_categorical

from dm_utils.utils.base import get_model_mode, get_log_level


def set_params(model, epochs=1000, lr=0.01, eval_rounds=100, early_stop_rounds=200, log_level=0, seed=42, num_classes=None):
    if isinstance(model, list):
        for m in model:
            set_params(m, epochs, eval_rounds, early_stop_rounds, log_level, seed)
    else:
        mode1, mode2 = get_model_mode(model)
        if mode1 == 'sklearn':
            if mode2 == 'sklearn':
                if isinstance(model, BaseDecisionTree):
                    model.set_params(random_state=seed)
                elif isinstance(model, BaseEnsemble):
                    model.set_params(n_estimators=epochs, random_state=seed)
            elif mode2 == 'xgboost':
                log_level = get_log_level('xgb', log_level)
                model.set_params(n_estimators=epochs, learning_rate=lr, early_stopping_rounds=early_stop_rounds, verbosity=log_level)
            elif mode2 == 'lightgbm':
                log_level = get_log_level('lgb', log_level)
                model.set_params(n_estimators=epochs, learning_rate=lr, verbosity=log_level, radom_state=seed)
            elif mode2 == 'catboost':
                log_level = get_log_level('cb', log_level)
                model.set_params(iterations=epochs, learning_rate=lr, logging_level=log_level, random_state=seed)
            elif mode2 == 'ngboost':
                if num_classes is not None:
                    model = model(Dist=k_categorical(num_classes))
                model.set_params(n_estimators=epochs, learning_rate=lr, verbose_eval=eval_rounds, early_stopping_rounds=early_stop_rounds, verbose=True, radom_state=seed)
            elif mode2 == 'tabnet':
                model.set_params(optimizer_params={'lr': lr}, verbose=eval_rounds, seed=seed)

    return model
