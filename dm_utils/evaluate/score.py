import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def logits2prob(logits):
    logits = np.array(logits).reshape(logits.shape[0], -1)
    if logits.shape[1] == 1:
        logits = np.hstack([-logits, logits])
        return softmax(logits, axis=1)[:, 1]
    else:
        return softmax(logits, axis=1)


def logits2label(logits):
    logits = np.array(logits).reshape(logits.shape[0], -1)
    if logits.shape[1] == 1:
        return (logits > 0).astype('int')
    else:
        return logits.argmax(axis=1)


def prob2label(proba):
    proba = np.array(proba).reshape(proba.shape[0], -1)
    if proba.shape[1] == 1:  # 二分类（只有类别 1 的概率）
        return (proba > 0.5).astype('int')
    else:  # 二分类（有类别 0、1 的概率）、多分类
        return proba.argmax(axis=1)


# 计算分数的方法
_score_name2func = {
    'acc': lambda t, p: accuracy_score(t, prob2label(p)),  # label, label
    'auc': roc_auc_score,  # (bcls)label, proba
    'ovr-auc': lambda t, p: roc_auc_score(t, p, multi_class='ovr'),  # (mcls，对类别不平衡比较敏感)label, proba
    'ovo-auc': lambda t, p: roc_auc_score(t, p, multi_class='ovo'),  # (mcls，对类别不平衡不敏感)label, proba
    'f1': f1_score,  # (bcls)label, label
    'micro-f1': lambda t, p: f1_score(t, prob2label(p), average='micro'),  # (mcls，计算总体 TP，再计算 F1)label, label
    'macro-f1': lambda t, p: f1_score(t, prob2label(p), average='macro'),  # (mcls，各类别 F1 的权重相同)label, label
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'r2': r2_score,
}


def get_score_names_funcs(score_names='', score_funcs=None, task=None):
    """根据 score_names, score_funcs 获取 score_names, score_funcs

    返回 score_names, score_funcs 都存在的计算分数方法

    Args:
        score_names (list): 逗号分隔的分数名称，如 'acc,auc'
        score_funcs (list): 计算分数的方法列表
        task (str): 任务类型, None, 'cls' 或 'reg'

    Returns:
        score_names (list): 计算分数的方法名称列表
        score_funcs (list): 计算分数的方法列表
    """
    # 如果不提供 score_names, score_funcs
    if score_names == '' and score_funcs is None:
        assert task is not None, 'task must be provided when score_names and score_funcs are not provided, but got None. task must be in [None, "cls", "reg"]'
        score_names = ['acc' if task == 'cls' else 'mse']
        score_funcs = [_score_name2func[name] for name in score_names]
    # 如果只提供 score_names
    elif score_names != '' and score_funcs is None:
        score_names = score_names.replace(' ', '').split(',')
        score_funcs = [_score_name2func[name] for name in score_names]
    # 如果只提供 score_funcs
    elif score_names == '' and score_funcs is not None:
        score_names = [func.__name__ for func in score_funcs]
    # 如果同时提供 score_names, score_funcs
    else:
        n1 = score_names.replace(' ', '').split(',')
        n2 = [func.__name__ for func in score_funcs]
        f1 = [_score_name2func[name] for name in n1]
        f2 = score_funcs
        score_names = n1 + n2
        score_funcs = f1 + f2

    return score_names, score_funcs
