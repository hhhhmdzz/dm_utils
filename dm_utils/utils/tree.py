import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List

from .base import get_model_mode


def get_feature(model, feature=None, importance_type='gain') -> np.array:
    """

    Parameters
    ----------
    model: trained model
    feature: feature name
    importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'

    Returns
    -------
    feature: feature name
    """
    mode1, mode2 = get_model_mode(model)
    if mode1 == 'lightgbm':
        feature = model.feature_name()
    elif mode1 == 'xgboost':
        feature = list(model.get_score(importance_type=importance_type).keys())
    elif mode2 == 'catboost':
        feature = model.feature_names_
    elif mode1 == 'sklearn':
        if mode2 in {'sklearn', 'xgboost'}:
            feature = model.feature_names_in_  # array
        elif mode2 == 'lightgbm':
            feature = model.feature_name_
        else:  # {'ngboost', 'tabnet'}
            assert feature is not None, 'feature should be provided for ngboost and tabnet'
    else:
        raise NotImplementedError

    return np.array(feature)


def get_importance(model, importance_type='gain', ngb=0) -> np.array:
    """

    Parameters
    ----------
    model: trained model
    importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
    ngb: 0 for loc trees, 1 for scale trees in ngboost

    Returns
    -------
    importance: feature importance
    """
    mode1, mode2 = get_model_mode(model)
    if mode1 == 'lightgbm':
        importance = model.feature_importance(importance_type=importance_type)
    elif mode1 == 'xgboost':
        importance = np.array(list(model.get_score(importance_type=importance_type).values()))
    elif mode1 == 'catboost':
        importance = model.get_feature_importance()
    elif mode1 == 'sklearn':
        importance = model.feature_importances_
        if mode2 == 'ngboost':
            importance = importance[ngb]
    else:
        raise NotImplementedError
    
    return importance


def get_feature_importance(
    feature: Union[np.array, List[np.array]],
    importance: Union[np.array, List[np.array]],
    sort=True, ascending=False, reduce=None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """

    Parameters
    ----------
    feature: feature name
    importance: feature importance
    sort: whether to sort
    ascending: whether to sort ascending
    reduce: how to reduce importance if feature is list

    Returns
    -------
    feature_importance: if feature is pd.DataFrame, then feature as index, importance as data

    Notes
    -----
    If feature is list, importance should also be list, and output feature_importance will be list.
    Further, if reduce is not None, feature_importance will be reduced to one.
    """
    f_is_list = isinstance(feature, list)
    i_is_list = isinstance(importance, list)
    assert f_is_list == i_is_list

    if f_is_list:
        feature_importances = [get_feature_importance(f, i, sort=sort, ascending=ascending) for f, i in zip(feature, importance)]
        if reduce is not None:
            feature_importances = [feature_importance.set_index('feature') for feature_importance in feature_importances]
            feature_importances = pd.DataFrame(pd.concat(feature_importances, axis=1).agg(reduce, axis=1), columns=['importance']).reset_index()
            if sort:
                feature_importances.sort_values(by='importance', ascending=ascending, inplace=True, ignore_index=True)
        return feature_importances
    else:
        feature_importance = pd.DataFrame((feature, importance)).rename({0: 'feature', 1: 'importance'}).T
        if sort:
            feature_importance.sort_values(by='importance', ascending=ascending, inplace=True, ignore_index=True)
        return feature_importance


def get_feature_importance_from_model(
    model, features=None, importance_type='gain', ngb=0, sort=True, ascending=False, reduce=None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """

    Parameters
    ----------
    model: trained model
    features: feature name
    importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
    ngb: 0 for loc trees, 1 for scale trees in ngboost
    sort: whether to sort
    ascending: whether to sort ascending
    reduce: how to reduce importance if feature is list

    Returns
    -------
    feature_importance: if feature is pd.DataFrame, then feature as index, importance as data

    Notes
    -----
    If model is list, feature_importance will be list.
    Further, parameter reduce, whcih is often used for k-fold models, turn the feature importance of k-fold models into one.
    """
    models = model if isinstance(model, list) else model
    if isinstance(models, list):
        features = [get_feature(model, features, importance_type=importance_type) for model in models]
        importances = [get_importance(model, importance_type=importance_type, ngb=ngb) for model in models]
    else:
        features = get_feature(models, features, importance_type=importance_type)
        importances = get_importance(models, importance_type=importance_type, ngb=ngb)
    feature_importances = get_feature_importance(features, importances, sort=sort, ascending=ascending, reduce=reduce)
    return feature_importances


def plot_feature_importance(
    features_importance, topk=20,
    save_img=False, save_path='feature_importance.png',
    show_img=True,
):
    """
    
    Parameters
    ----------
    features_importance: feature importance
    topk: top k features
    save_img: whether to save image
    save_path: path to save image
    show_img: whether to show image

    Returns
    -------
    None
    """
    features_importance = features_importance.iloc[:topk, :]
    L = len(features_importance)
    width = 10
    height = L // 3 if L < 30 else L // 5
    plt.figure(figsize=(width, height), dpi=100)
    sns.barplot(
        data=features_importance, y='feature', x='importance',
        orient='horizontal', palette='husl',
    )
    plt.title('Feature Importance')
    plt.ylabel(f'{L} features')
    plt.xlabel('importances')

    if save_img:
        plt.savefig(save_path)

    if show_img:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    pass
