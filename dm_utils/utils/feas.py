import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from typing import Optional

from .tree import get_feature_importance_from_model
from .cprint import info, success, warning


def cont_var_dist(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame] = None,
    titles: Optional[list[str]] = None,
    ncol: Optional[int] = None,
    plot_kde=True, plot_hist=True,
    hist_multiple='stack',
    show_img=True, save_img=False, img_name=None,
    return_figs=False,
):
    feas = list(train.columns)
    feas_num = train.shape[1]
    titles = range(feas_num) if titles is None else titles

    data = pd.concat((train, test), axis=0) if test is not None else train.copy()
    data = data.reset_index(drop=True)
    data['is_train'] = data.index < len(train)

    figs = []
    ncol = int(np.ceil(feas_num**.5)) if ncol is None else ncol
    for i, fea in enumerate(feas):
        if i % ncol**2 == 0:
            plt.figure(figsize=(4 * ncol, 4 * ncol), dpi=100)
        plt.subplot(ncol, ncol, i % ncol**2 + 1)
        plt.title(f'[{titles[i]}]{feas[i]}')
        if plot_kde:
            ax = sns.kdeplot(data.query('is_train == 1')[fea], shade=True, color='r', label='train')
            ax = sns.kdeplot(data.query('is_train == 0')[fea], shade=True, color='b', label='test')
            # ax = sns.kdeplot(data.query('is_train == 1')[fea], fill=True, color='r', label='train')  # new version
            # ax = sns.kdeplot(data.query('is_train == 0')[fea], fill=True, color='b', label='test')  # new version
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend(loc='upper left')
        else:
            plt.yticks([])
        if plot_hist:
            ax2 = plt.twinx()
            ax2 = sns.histplot(data=data, x=fea, hue='is_train', bins=10, multiple=hist_multiple, legend=True, alpha=0.25)
            ax2.set_xlabel('')
            ax2.set_ylabel('')
            ax2.legend(['train', 'test'], loc='upper right')
        if (i % ncol**2 == ncol**2 - 1) or (i == feas_num - 1):
            if return_figs:
                figs.append(plt.gcf())
            if save_img:
                plt.savefig(img_name)
            if show_img:
                plt.show()
            else:
                plt.close()
    if return_figs:
        return figs
    return None


def disc_var_dist(var1, var2, target, var1_name=None, var2_name=None, target_name=None):
    """
    discrete variable distribution

    Parameters
    ----------
    var1: pd.Series or np.array
    var2: pd.Series or np.array
    target: pd.Series or np.array
    var1_name: str
    var2_name: str
    target_name: str
    """
    unique_var1 = np.unique(var1)
    unique_var2 = np.unique(var2)
    unique_target = np.unique(target)

    if isinstance(var1, (pd.Series, np.array)):
        var1_name = var1_name or var1.name or 'var1'
    else:
        raise TypeError("var1 must be pd.Series or np.array")

    if isinstance(var2, (pd.Series, np.array)):
        var2_name = var2_name or var2.name or 'var2'
    else:
        raise TypeError("var2 must be pd.Series or np.array")

    if isinstance(target, (pd.Series, np.array)):
        target_name = target_name or target.name or 'target'
    else:
        raise TypeError("target must be pd.Series or np.array")

    html = "<table>" + \
    "<tr><td>{" + "}</td><td>{" * len(unique_target) + "}</td></tr>" + \
    "<tr><td>{" + "}</td><td>{" * len(unique_target) + "}</td></tr>" + \
    "<tr><td>{" + "}</td><td>{" * len(unique_target) + "}</td></tr>" + \
    "</table>"
    data_target = [''] + [f'{target_name} = {t}, rate = {sum(target == t) / len(target)} = {sum(target == t)} / {len(target)}' for t in unique_target]
    data_num = ['num']
    data_rate = ['rate']
    for t in unique_target:
        df = pd.DataFrame(np.zeros((len(unique_var1), len(unique_var2)), dtype=int), index=unique_var1, columns=unique_var2)
        df.index.name = var1_name
        df.columns.name = var2_name
        for i, v1 in enumerate(unique_var1):
            for j, v2 in enumerate(unique_var2):
                df.iloc[i, j] = ((var1 == v1) & (var2 == v2) & (target == t)).sum()
        data_num.append(df.to_html())
        data_rate.append((100 * df / df.sum().sum()).round(3).to_html())

    display(HTML(html.format(*data_target, *data_num, *data_rate)))


def adversarial_validation(
    train, test, clf=None, seed=42,
    softmax_feature_importance=False,
    return_feature_importance=False
):
    """
    Adversarial validation
    
    Parameters
    ----------
    train: pd.DataFrame
    test: pd.DataFrame
    clf: sklearn.base.BaseEstimator
    seed: int
    softmax_feature_importance: bool
    return_feature_importance: bool

    Returns
    -------
    auc: float
    train_proba: np.array, advertarial validation scores on train set
    feature_importance[Optional]: pd.DataFrame
    """
    assert train.shape[1] == test.shape[1], "train and test must have the same number of features"

    data = pd.concat([train, test])
    data['is_train'] = [1] * len(train) + [0] * len(test)
    xtrain, xtest, ytrain, ytest = train_test_split(data.drop(columns=['is_train']), data['is_train'], test_size=0.2, random_state=seed)

    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(xtrain, ytrain)
    train_proba = clf.predict_proba(train)[:, 1]
    auc = roc_auc_score(ytest, clf.predict_proba(xtest)[:, 1])

    if return_feature_importance:
        try:
            feature_importance = get_feature_importance_from_model(clf)
            if softmax_feature_importance:
                feature_importance['importance'] = softmax(feature_importance['importance'])
            return auc, train_proba, feature_importance
        except:
            warning("Cannot get feature importance from model.")
            return auc, train_proba, None
    return auc, train_proba


def adversarial_validation_features(
    train, test, clf=None,
    drop_n=1, drop_rate=None, drop_thresold=None,
    seed=42,
    softmax_feature_importance=False,
    return_feature_importance=False,
):
    """
    Adversarial validation with feature selection

    Parameters
    ----------
    train: pd.DataFrame
    test: pd.DataFrame
    clf: sklearn.base.BaseEstimator
    drop_n: , drop number of features
    drop_rate: float, drop rate of features
    drop_thresold: float, drop threshold of feature importance
    seed: int
    softmax_feature_importance: bool
    return_feature_importance: bool

    Returns
    -------
    auc: float
    train_proba: np.array, advertarial validation scores on train set
    drop_feas: list, features to be dropped
    remain_feas: list, features to be remained
    feature_importance[Optional]: pd.DataFrame

    Notes
    ----
    Order of priority: drop_n > drop_rate > drop_thresold
    """
    num_feas = train.shape[1]
    auc, train_proba, feature_importance = adversarial_validation(
        train, test, clf, seed, return_feature_importance=True,
        softmax_feature_importance=softmax_feature_importance)

    if drop_n is not None:
        num_drop_feas = max(1, drop_n)
    elif drop_rate is not None:
        num_drop_feas = max(1, int(num_feas * drop_rate))
    elif drop_thresold is not None:
        num_drop_feas = max(1, min(num_drop_feas, (feature_importance['importance'] > drop_thresold).sum()))
    else:
        num_drop_feas = 1
    
    drop_feas = feature_importance['feature'][:num_drop_feas].tolist()
    remain_feas = feature_importance['feature'][num_drop_feas:].tolist()

    if return_feature_importance:
        return auc, train_proba, drop_feas, remain_feas, feature_importance
    return auc, train_proba, drop_feas, remain_feas


def adversarial_validation_instances(
    train, test, clf=None,
    select_rate=0.2, select_n=None, select_thresold=None,
    seed=42,
    return_data_ins=False,
):
    """
    Adversarial validation with instance selection

    Parameters
    ----------
    train: pd.DataFrame
    test: pd.DataFrame
    clf: sklearn.base.BaseEstimator
    select_rate: float, select rate of instances
    select_n: int, select number of instances
    select_thresold: float, select threshold of instance probability
    seed: int
    return_data_ins: bool

    Returns
    -------
    auc: float
    train_proba: np.array, advertarial validation scores on train set
    select_idx: list, indices of selected instances
    remain_idx: list, indices of remained instances
    select_ins[Optional]: pd.DataFrame
    remain_ins[Optional]: pd.DataFrame
    """
    num_ins = train.shape[0]
    auc, train_proba = adversarial_validation(train, test, clf, seed, return_feature_importance=False)

    if select_n is not None:
        num_select_ins = max(1, select_n)
    elif select_rate is not None:
        num_select_ins = max(1, int(num_ins * select_rate))
    elif select_thresold is not None:
        num_select_ins = max(1, min(num_select_ins, (train_proba > select_thresold).sum()))
    else:
        num_select_ins = int(num_ins * 0.2)

    select_idx = np.argsort(train_proba)[-num_select_ins:].tolist()
    remain_idx = np.argsort(train_proba)[:-num_select_ins].tolist()
    select_ins = train.iloc[select_idx]  # valid
    remain_ins = train.iloc[remain_idx]  # train

    if return_data_ins:
        return auc, train_proba, select_idx, remain_idx, (select_ins, remain_ins)
    return auc, train_proba, select_idx, remain_idx
