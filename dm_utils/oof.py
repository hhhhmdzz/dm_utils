import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, Union

import dm_utils.runner as u_runner
# import dm_utils.evaluate as u_eval
from dm_utils.evaluate import score as ue_score
import dm_utils.task as u_task
# import dm_utils.utils as u_utils
from dm_utils.utils import cprint as uu_print
from dm_utils.utils import base as uu_base
from dm_utils.utils import feas as uu_feas
from dm_utils.utils import tree as uu_tree


class OOF(BaseEstimator):
    """
    OOF: Out-Of-Fold

    Parameters
    ----------
    task: reg, cls
    model: model, str, list of model, list of str
    kfold: kfold, gkfold, skfold, sgkfold
    folds: number of folds
    groups: groups
    score_names: score names
    score_funcs: score functions
    sklearn_api: whether use sklearn api
    seed: random seed
    """
    def __init__(
        self, task, model,
        kfold='kfold', folds=5, groups=None,
        epochs=1000, eval_rounds=100, early_stop_rounds=200, log_level=0,
        score_names='', score_funcs=None,
        sklearn_api=False,
        seed=42,
    ):
        self._task = task
        self.model = None  # untrained model(s)
        self.kfold = kfold
        self.folds = folds
        self.groups = groups
        self.seed = seed
        
        self.epochs = epochs
        self.eval_rounds = eval_rounds
        self.early_stop_rounds = early_stop_rounds
        self.log_level = log_level

        self.score_names, self.score_funcs = ue_score.get_score_names_funcs(score_names, score_funcs, task)

        self.models = []  # trained models
        self._scores = {}
        self.num_classes = None

        self.is_sep_model = isinstance(model, list)  # is separate models
        if self.is_sep_model:
            assert len(model) == folds, f'len(model) must equal to folds if model is a list, but len(model) == {len(model)} != folds == {folds}'
            self.model = [u_task.get_model_from_str(task, model[i], sklearn_api) if isinstance(model[i], str) else model[i] for i in range(folds)]
            self.model_name = [uu_base.get_model_name(m) for m in self.model]
        else:
            self.model = u_task.get_model_from_str(task, model, sklearn_api) if isinstance(model, str) else model
            self.model_name = uu_base.get_model_name(self.model)
        self.all_model_name = ','.join(np.unique(self.model_name).tolist()) if self.is_sep_model else self.model_name

    def fit(self, X_train, y_train, record_time=True, **kwargs):
        self.models = []
        self.feature = X_train.columns.tolist()
        self.indexes = np.arange(len(X_train))
        np.random.RandomState(self.seed).shuffle(self.indexes)

        self.data_size = len(X_train)
        self.valid_size = len(X_train) // self.folds
        self.train_size = self.data_size - self.valid_size
        self.kf = self._get_kfold(X_train, y_train, train_size=self.train_size, groups=self.groups)

        self.num_classes = len(np.unique(y_train)) if self._task == 'cls' else -1

        t_fit_0 = time()
        uu_print.info(f'{self.folds}-fold training begin.')
        if self._task == 'reg' or (self._task == 'cls' and self.num_classes == 2):
            self.y_train_pred = np.zeros(len(X_train))
        else:
            self.y_train_pred = np.zeros((len(X_train), self.num_classes))
        self._scores = {}  # Dict[Dict[str, Union[str, float]]], {'foldi': {'model': model_name, 'score1': score1, 'score2': score2, ...}}
        for i, (trn_idx, val_idx) in enumerate(self.kf):
            t_fold_0 = time()
            model = self.model[i] if self.is_sep_model else self.model
            model_name = self.model_name[i] if self.is_sep_model else self.model_name

            uu_print.info(f"Model {model_name}, Fold {i+1} / {self.folds} training begin.")
            X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
            y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
            model = u_runner.train(
                self._task, model, X_trn, y_trn, X_val, y_val,
                epochs=self.epochs, eval_rounds=self.eval_rounds, early_stop_rounds=self.early_stop_rounds, log_level=self.log_level)
            self.models.append(deepcopy(model))
            self.y_train_pred[val_idx] = u_runner.predict(self._task, model, X_val)
            val_scores = [func(y_val, self.y_train_pred[val_idx]) for func in self.score_funcs]
            val_scores = dict(zip(self.score_names, val_scores))
            val_scores['model'] = model_name  # Dict[str, Union[str, float]], {'model': model_name, 'score1': score1, 'score2': score2, ...}
            self._scores[f'fold{i}'] = val_scores

            t_fold = time() - t_fold_0
            if record_time: uu_print.info(f'Model {model_name}, Fold {i+1} / {self.folds} training finish, cost time {round(t_fold, 3)} s.')
            else: uu_print.info(f'Model {model_name}, Fold {i+1} / {self.folds} training finish.')
            uu_print.success(f'{i+1} / {self.folds} fold validation scores: {val_scores}')

        valid_scores = [func(y_train, self.y_train_pred) for func in self.score_funcs]
        valid_scores = dict(zip(self.score_names, valid_scores))
        valid_scores['model'] = self.all_model_name  # Dict[str, Union[str, float]], {'model': model_name, 'score1': score1, 'score2': score2, ...}
        self._scores['all'] = valid_scores

        t_fit = time() - t_fit_0
        if record_time: uu_print.info(f'{self.folds}-fold training finish, cost time {round(t_fit, 3)} s.')
        else: uu_print.info(f'{self.folds}-fold training finish.')
        uu_print.success(f'total {self.folds}-fold validation scores:{valid_scores}')

        self._scores = self.get_scores()
        return self._scores

    def score(self, X, y, y_pred=None, as_label=True, metric=None):
        if y_pred is None:
            y_pred = self.predict(X)
            if self._task == 'cls' and as_label:
                y_pred = ue_score.prob2label(y_pred)
        if metric is None:
            metric = accuracy_score if self._task == 'cls' else mean_squared_error
        return metric(y, y_pred)

    def predict(self, X):
        assert len(self.models) != 0, 'Model is not trained yet.'

        if self._task == 'reg' or (self._task == 'cls' and self.num_classes == 2):
            self.y_pred = np.zeros(len(X))
        else:
            self.y_pred = np.zeros((len(X), self.num_classes))

        for model in self.models:
            self.y_pred += u_runner.predict(self._task, model, X) / self.folds
        
        return self.y_pred

    def _get_kfold(self, X_train, y_train, train_size, shuffle=True, groups=None):
        n_splits = self.folds
        groups = [i for i in range(1, n_splits+1)] * (train_size // n_splits) + [i for i in range(1, train_size % n_splits + 1)] if groups is None else groups
        
        if self.kfold == 'kfold':
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed).split(X_train, y_train)
        elif self.kfold == 'gkfold':
            kf = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed).split(X_train, y_train, groups)
        elif self.kfold == 'skfold':
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed).split(X_train, y_train)
        elif self.kfold == 'sgkfold':
            kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed).split(X_train, y_train, groups)
        else:
            raise ValueError('Unknown kfold type')

        return kf

    @property
    def task(self):
        return self._task

    def get_scores(self) -> pd.DataFrame:
        assert len(self._scores) > 0, 'Model is not trained yet.'
        return pd.DataFrame(self._scores).T if isinstance(self._scores, dict) else self._scores

    @property
    def scores(self):
        return self.get_scores()

    def get_feature_importance(self, fold=None, sort=True, ascending=False, reduce='sum'):
        assert len(self.models) != 0, 'Model is not trained yet.'
        return uu_tree.get_feature_importance_from_model(
            self.models[fold] if fold is not None else self.models,
            self.feature, sort=sort, ascending=ascending, reduce=reduce
        )

    def feature_importances(self):
        return self.get_feature_importance(reduce=None)

    def feature_importance(self, fold=None, reduce='sum'):
        return self.get_feature_importance(fold=fold, reduce=reduce)


if __name__ == '__main__':
    pass
