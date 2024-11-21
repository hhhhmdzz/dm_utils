# README

`dm_utils` is a utility for Data Mining.

## Installation

```bash
pip install dm_utils
```

## Usage

- `dm_utils.hom` : hold-out method

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from dm_utils.oof import OOF

x, y = load_iris(return_X_y=True, as_frame=True)
# classification task, xgboost and lightgbm model
oof = OOF(task='cls', model=['xgb', 'lgb'])
oof.fit(xtrain, ytrain, record_time=True)
ypred = (oof.predict(xtest) > 0.5).argmax(axis=1)
print(accuracy_score(ypred, ytest))
```

- `dm_utils.oof` : out of fold prediction

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from dm_utils.oof import OOF

x, y = load_breast_cancer(return_X_y=True, as_frame=True)
# classification task, 2***xgboost**, 2*lightgbm and catboost model for 5-fold oof
oof = OOF(task='cls', model=['xgb', 'xgb', 'lgb', 'lgb', 'cb'])
oof.fit(xtrain, ytrain, record_time=True)
ypred = oof.predict(xtest) > 0.5
print(accuracy_score(ypred, ytest))
```

## Features

support algorithm: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `ngboost` and `pytorch-tabnet`
