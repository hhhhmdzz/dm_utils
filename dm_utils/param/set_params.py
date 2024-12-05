import inspect
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
from pytorch_tabnet.abstract_model import TabModel
from sklearn.base import BaseEstimator
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import BaseEnsemble
from ngboost.distns import k_categorical

from dm_utils.utils.base import get_model_mode, get_log_level


def set_params(
    model,
    epochs=1000,
    lr=0.01,
    eval_rounds=100,
    early_stop_rounds=200,
    log_level=0,
    seed=42,
    num_classes=None,
    params=None,
):
    """
    set params for model

    Parameters
    ----------
    model: model
    epochs: int
    lr: float
    eval_rounds: int
    early_stop_rounds: int
    log_level: int
    seed: int
    num_classes: int
    params: dict

    Returns
    -------
    model: model
    """
    if isinstance(model, list) and isinstance(params, list):
        for m, p in zip(model, params):
            set_params(m, epochs, eval_rounds, early_stop_rounds, log_level, seed, p)
    else:
        params = params or {}
        mode1, mode2 = get_model_mode(model)
        if mode1 == 'sklearn':
            if mode2 == 'sklearn':
                if isinstance(model, BaseDecisionTree):
                    model.set_params(random_state=seed, **params)
                elif isinstance(model, BaseEnsemble):
                    model.set_params(
                        n_estimators=epochs,
                        random_state=seed,
                        **params
                    )
                elif isinstance(model, BaseEstimator):
                    sig = inspect.signature(model.set_params)
                    if 'random_state' in sig.parameters.keys():
                        model.set_params(random_state=seed)
                    model.set_params(**params)
            elif isinstance(model, BaseEstimator) \
                and isinstance(model, xgb.XGBModel):  # mode2 == 'xgboost'
                log_level = get_log_level('xgb', log_level)
                model.set_params(
                    n_estimators=epochs,
                    learning_rate=lr,
                    early_stopping_rounds=early_stop_rounds,
                    verbosity=log_level,
                    **params
                )
            elif isinstance(model, lgb.LGBMModel):  # mode2 == 'lightgbm'
                log_level = get_log_level('lgb', log_level)
                model.set_params(
                    n_estimators=epochs,
                    learning_rate=lr,
                    verbosity=log_level,
                    radom_state=seed,
                    **params
                )
            elif isinstance(model, cb.CatBoost):  # mode2 == 'catboost'
                log_level = get_log_level('cb', log_level)
                model.set_params(
                    iterations=epochs,
                    learning_rate=lr,
                    logging_level=log_level,
                    random_state=seed,
                    feature_weights=None,
                    **params
                )
            elif type(model) is type and issubclass(model, ngb.NGBoost):  # mode2 == 'ngboost'
                if num_classes is not None:
                    model = model(Dist=k_categorical(num_classes))
                model.set_params(
                    n_estimators=epochs,
                    learning_rate=lr,
                    verbose_eval=eval_rounds,
                    early_stopping_rounds=early_stop_rounds,
                    verbose=True,
                    radom_state=seed,
                    # natural_gradient=False,
                    **params
                )
            elif isinstance(model, TabModel):  # mode2 == 'tabnet'
                model.set_params(
                    optimizer_params={'lr': lr},
                    verbose=eval_rounds,
                    seed=seed,
                    **params
                )

    return model
