import numpy as np
import pandas as pd
from IPython.display import display
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

from .tree import get_feature_importance_from_model
from .cprint import info, success


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
