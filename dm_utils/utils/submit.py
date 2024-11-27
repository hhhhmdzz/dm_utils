import os
import os.path as osp
import numpy as np
import pandas as pd
from IPython.display import display
from IPython.core.display import HTML

from ..evaluate.score import get_score_names_funcs


def cmp_classification(submissions, scores, names=None, score_func=None, show_res=True, return_res=True):
    L = len(submissions)
    assert L == len(scores)

    if score_func is None:
        _, score_func = get_score_names_funcs(score_names='acc', task='cls')
        score_func = score_func[0]

    df_cmp = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            df_cmp[i, j] = score_func(submissions[i], submissions[j])

    df_scores = pd.DataFrame(scores, index=names, columns=['scores'])
    df_cmp = pd.DataFrame(df_cmp, index=names, columns=names)

    if show_res:
        for i in range(0, L, 10):
            html = ("<table>" + \
            "<tr><td></td><td>{}</td></tr>" + \
            "<tr><td>{}</td><td>{}</td></tr>" + \
            "</table>").format(
                df_scores.T.style.hide(axis=0).to_html(),
                df_scores.iloc[i:i+10].style.hide(axis=1).to_html(),
                df_cmp.iloc[i:i+10].style.hide(axis=0).hide(axis=1).to_html(),
            )
            display(HTML(html))
            # display(df_cmp.iloc[i:i+10])

    if return_res:
        return df_cmp


def cmp_regression(submissions, scores, names=None, score_func=None, show_res=True, return_res=True):
    L = len(submissions)
    assert L == len(scores)

    if score_func is None:
        _, score_func = get_score_names_funcs(score_names='mse', task='reg')
        score_func = score_func[0]

    df_cmp = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            df_cmp[i, j] = score_func(submissions[i], submissions[j])

    df_scores = pd.DataFrame(scores, index=names, columns=['scores'])
    df_cmp = pd.DataFrame(df_cmp, index=names, columns=names)

    if show_res:
        for i in range(0, L, 10):
            html = ("<table>" + \
            "<tr><td></td><td>{}</td></tr>" + \
            "<tr><td>{}</td><td>{}</td></tr>" + \
            "</table>").format(
                df_scores.T.style.hide(axis=0).to_html(),
                df_scores.iloc[i:i+10].style.hide(axis=1).to_html(),
                df_cmp.iloc[i:i+10].style.hide(axis=0).hide(axis=1).to_html(),
            )
            display(HTML(html))
            # display(df_cmp.iloc[i:i+10])

    if return_res:
        return df_cmp

