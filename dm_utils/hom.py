import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

import dm_utils.runner as u_runner
# import dm_utils.evaluate as u_eval
from dm_utils.evaluate import score as ue_score
import dm_utils.task as u_task
# import dm_utils.utils as u_utils
from dm_utils.utils import cprint as uu_print
from dm_utils.utils import base as uu_base
from dm_utils.utils import feas as uu_feas
from dm_utils.utils import tree as uu_tree


class HOM(BaseEstimator):
    """
    HOM: Hold-Out Method
    """
    def __init__(
        self, task, model,
        epochs=1000, eval_rounds=100, early_stop_rounds=200, log_level=0,
        score_names='', score_funcs=None,
        sklearn_api=False,
        seed=42,
    ):
        self._task = task
        self.model = None  # untrained model(s)
        self.seed = seed
        
        self.epochs = epochs
        self.eval_rounds = eval_rounds
        self.early_stop_rounds = early_stop_rounds
        self.log_level = log_level

        self.score_names, self.score_funcs = ue_score.get_score_names_funcs(score_names, score_funcs, task)

        self.models = []  # trained model(s)
        self._scores = {}
        self.num_classes = None

        self.is_sep_model = isinstance(model, list)  # is separate models
        if self.is_sep_model:
            self.model = [u_task.get_model_from_str(task, m, sklearn_api) if isinstance(m, str) else m for m in model]
            self.model_name = [uu_base.get_model_name(m) for m in self.model]
        else:
            self.model = u_task.get_model_from_str(task, model, sklearn_api) if isinstance(model, str) else model
            self.model_name = uu_base.get_model_name(self.model)
        self.num_models = len(self.model) if self.is_sep_model else 1
        self.all_model_name = ','.join(np.unique(self.model_name).tolist()) if self.is_sep_model else self.model_name

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, test_size=0.2, record_time=True, **kwargs):
        self.models = [] if self.is_sep_model else None
        self.feature = X_train.columns.tolist()
        self.indexes = np.arange(len(X_train))
        np.random.RandomState(self.seed).shuffle(self.indexes)

        if X_valid is None and y_valid is None:
            uu_print.info('not provided X_valid and y_valid, auto split train set into train and valid.')
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=self.seed)

        self.train_size = len(X_train)
        self.valid_size = len(X_valid)

        self.num_classes = len(np.unique(y_train)) if self._task == 'cls' else -1

        t_fit_0 = time()
        uu_print.info(f'hold-out method training begin.')
        if self._task == 'reg' or (self._task == 'cls' and self.num_classes == 2):
            self.y_valid_pred = np.zeros(len(X_valid))
        else:
            self.y_valid_pred = np.zeros((len(X_valid), self.num_classes))
        self._scores = {}  # Dict[Dict[str, Union[str, float]]], {'modeli': {'model': model_name, 'score1': score1, 'score2': score2, ...}}
        for i, model in enumerate(self.model if self.is_sep_model else [self.model]):
            t_fold_0 = time()
            model_name = self.model_name[i] if self.is_sep_model else self.model_name

            uu_print.info(f"Model {model_name} {i+1} / {self.num_models} training begin.")
            model = u_runner.train(
                self._task, model, X_train, y_train, X_valid, y_valid,
                epochs=self.epochs, eval_rounds=self.eval_rounds, early_stop_rounds=self.early_stop_rounds, log_level=self.log_level)
            if self.is_sep_model:
                self.models.append(deepcopy(model))
            else:
                self.models = deepcopy(model)
            _pred = u_runner.predict(self._task, model, X_valid)
            self.y_valid_pred += _pred / self.num_models
            val_scores = [func(y_valid, _pred) for func in self.score_funcs]
            val_scores = dict(zip(self.score_names, val_scores))
            val_scores['model'] = model_name  # Dict[str, Union[str, float]], {'model': model_name, 'score1': score1, 'score2': score2, ...}
            self._scores[f'model{i}'] = val_scores

            t_fold = time() - t_fold_0
            if record_time: uu_print.info(f'Model {model_name} {i+1} / {self.num_models} training finish, cost time {round(t_fold, 3)} s.')
            else: uu_print.info(f'Model {model_name} {i+1} / {self.num_models} training finish.')
            uu_print.success(f'{i+1} / {self.num_models} model validation scores: {val_scores}')

        valid_scores = [func(y_valid, self.y_valid_pred) for func in self.score_funcs]
        valid_scores = dict(zip(self.score_names, valid_scores))
        valid_scores['model'] = self.all_model_name  # Dict[str, Union[str, float]], {'model': model_name, 'score1': score1, 'score2': score2, ...}
        self._scores['all'] = valid_scores

        t_fit = time() - t_fit_0
        if record_time: uu_print.info(f'hold-out method training finish, cost time {round(t_fit, 3)} s.')
        else: uu_print.info(f'hold-out method training finish.')
        uu_print.success(f'total {self.num_models} model validation scores:{valid_scores}')

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

    def predict(self, X, model_idx=None):
        assert len(self._scores), 'Model is not trained yet.'

        if self._task == 'reg' or (self._task == 'cls' and self.num_classes == 2):
            self.y_pred = np.zeros(len(X))
        else:
            self.y_pred = np.zeros((len(X), self.num_classes))

        if self.is_sep_model:
            if model_idx is None:
                for model in self.models:
                    self.y_pred += u_runner.predict(self._task, model, X) / self.num_models
            else:
                self.y_pred = u_runner.predict(self._task, self.models[model_idx], X)
        else:
            self.y_pred = u_runner.predict(self._task, self.models, X)
        
        return self.y_pred

    @property
    def task(self):
        return self._task

    def get_scores(self):
        assert len(self._scores) > 0, 'Model is not trained yet.'
        return pd.DataFrame(self._scores).T if isinstance(self._scores, dict) else self._scores

    @property
    def scores(self):
        return self.get_scores()

    def get_feature_importance(self, model_idx=None, sort=True, ascending=False):
        assert len(self._scores), 'Model is not trained yet.'
        return uu_tree.get_feature_importance_from_model(
            self.models[model_idx] if model_idx is not None else self.models,
            self.feature, sort=sort, ascending=ascending, reduce=None
        )

    def feature_importances(self):
        return self.get_feature_importance()
    
    def feature_importance(self, model_idx=None):
        assert not (self.is_sep_model and model_idx is None), 'model_idx must be specified.'
        return self.get_feature_importance(model_idx=model_idx)


if __name__ == '__main__':
    pass
