import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import ngboost as ngb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_iris
from ngboost.distns import k_categorical
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from ngboost import NGBClassifier, NGBRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import dm_utils as dmu

seed = 42
epochs = 100
eval_rounds = 10
early_stop_rounds = 20
log_level = 0

# ---------- regression ----------
x, y = fetch_california_housing(return_X_y=True, as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

xgb_gpu_params = dmu.param.get_gpu_params('xgb')
lgb_gpu_params = dmu.param.get_gpu_params('lgb')
cb_gpu_params = dmu.param.get_gpu_params('cb')

# sklearn api, gpu
for model in [
    DecisionTreeRegressor(random_state=seed),
    RandomForestRegressor(n_estimators=epochs, random_state=seed),
    XGBRegressor(n_estimators=epochs, early_stopping_rounds=early_stop_rounds, verbosity=dmu.utils.get_log_level('xgb', log_level), **xgb_gpu_params),
    LGBMRegressor(n_estimators=epochs, verbosity=dmu.utils.get_log_level('lgb', log_level), **lgb_gpu_params),
    CatBoostRegressor(iterations=epochs, logging_level=dmu.utils.get_log_level('cb', log_level), **cb_gpu_params),
    NGBRegressor(n_estimators=epochs, verbose_eval=eval_rounds, early_stopping_rounds=early_stop_rounds, verbose=True),
    TabNetRegressor(verbose=eval_rounds),
]:
    print('-'*50, model.__class__.__name__, '-'*50)
    model = dmu.runner.train(
        'reg', model, x_train, y_train, x_test, y_test, epochs=epochs, eval_rounds=eval_rounds,
        early_stop_rounds=early_stop_rounds, log_level=log_level, use_gpu4tree=True)
    pred = dmu.runner.predict('reg', model, x_test)
    print(pred.shape)

# original api, cpu
for model in [
    xgb.Booster,
    lgb.Booster,
    cb.CatBoost,
]:
    print('-'*50, model, '-'*50)
    model = dmu.runner.train(
        'reg', model, x_train, y_train, x_test, y_test, epochs=epochs, eval_rounds=eval_rounds,
        early_stop_rounds=early_stop_rounds, log_level=log_level, use_gpu4tree=False)
    pred = dmu.runner.predict('reg', model, x_test)
    print(pred.shape)

# ---------- classification ----------
x, y = load_iris(return_X_y=True, as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# sklearn api, cpu
for model in [
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(n_estimators=epochs, random_state=seed),
    XGBClassifier(n_estimators=epochs, early_stopping_rounds=early_stop_rounds, verbosity=dmu.utils.get_log_level('xgb', log_level)),
    LGBMClassifier(n_estimators=epochs, verbosity=dmu.utils.get_log_level('lgb', log_level)),
    CatBoostClassifier(iterations=epochs, logging_level=dmu.utils.get_log_level('cb', log_level)),
    NGBClassifier(n_estimators=epochs, verbose_eval=eval_rounds, verbose=True, Dist=k_categorical(len(np.unique(y_train)))),
    TabNetClassifier(verbose=eval_rounds),
]:
    print('-'*50, model.__class__.__name__, '-'*50)
    model = dmu.runner.train(
        'cls', model, x_train, y_train, x_test, y_test, epochs=epochs, eval_rounds=eval_rounds,
        early_stop_rounds=early_stop_rounds, log_level=log_level, use_gpu4tree=False)
    pred = dmu.runner.predict('cls', model, x_test)
    print(pred.shape)

# original api, gpu
for model in [
    xgb.Booster,
    lgb.Booster,
    cb.CatBoost,
]:
    print('-'*50, model, '-'*50)
    model = dmu.runner.train(
        'cls', model, x_train, y_train, x_test, y_test, epochs=epochs, eval_rounds=eval_rounds,
        early_stop_rounds=early_stop_rounds, log_level=log_level, use_gpu4tree=True)
    pred = dmu.runner.predict('cls', model, x_test)
    print(pred.shape)
