import os
import json

# file directory
root = os.path.dirname(__file__)

reg_path = os.path.join(root, 'reg.json')
reg_params = [
    {
        'loss_function': 'RMSE',
        'custom_metric': ['RMSE', 'MAE'],
        'eval_metric': 'MAE',

        # 'boosting_type': 'Auto',  # Auto Ordered Plain
        'learning_rate': 0.03,
        'iterations': 15000,  # num_boost_round
        'subsample': 0.66,
        'sampling_frequency': 'PerTreeLevel',  # PerTreeLevel PerTree
        # min_split_gain
        'min_data_in_leaf': 1,  # min_child_samples
        # min_child_weight
        'rsm': 0.5,  # colsample_bylevel
        'leaf_estimation_method': 'Gradient',  # Newton Gradient Exact
        'one_hot_max_size': 2,
        'fold_len_multiplier': 2,

        'depth': 10,
        'grow_policy': 'SymmetricTree',  # SymmetricTree, Depthwise, Lossguide
        'max_leaves': 31,  # num_leaves
        'bagging_temperature': 1,
        'random_strength': 1,
        'l2_leaf_reg': 18,  # reg_lambda
        # reg_alpha

        'early_stopping_rounds': 300,
        'od_type': 'Iter',  # IncToDec, Iter

        'border_count': 128,  # max_bin

        'random_seed': 42,
        'thread_count': -1,
        'use_best_model': True,
        'logging_level': 'Verbose',
    },
]

bcls_path = os.path.join(root, 'bcls.json')
bcls_params = [
    {
        'loss_function': 'Logloss',
        'custom_metric': ['F1', 'Accuracy', 'AUC'],
        'eval_metric': 'AUC',

        # 'boosting_type': 'Auto',  # Auto Ordered Plain
        'learning_rate': 0.03,
        'iterations': 15000,  # num_boost_round
        'scale_pos_weight': 1.0,
        'subsample': 0.66,
        # min_split_gain
        'min_data_in_leaf': 1,  # min_child_samples
        # min_child_weight
        'rsm': 1,
        'leaf_estimation_method': 'Gradient',  # Newton Gradient Exact
        'one_hot_max_size': 2,
        'fold_len_multiplier': 2,

        'depth': 6,
        'max_leaves': 31,  # num_leaves
        'bagging_temperature': 1,
        'random_strength': 1,
        'l2_leaf_reg': 3,  # reg_lambda
        # reg_alpha

        'border_count': 128,

        'random_seed': 42,
        'thread_count': -1,
        'use_best_model': True,
        'logging_level': 'Verbose',
    },
]

mcls_path = os.path.join(root, 'mcls.json')
mcls_params = [
    {
        'los`s_function': 'MultiClass',
        'custom_metric': ['MultiClass', 'Accuracy'],
        'eval_metric': 'MultiClass',

        # 'boosting_type': 'Auto',  # Auto Ordered Plain
        'learning_rate': 0.06503043695984914,
        'iterations': 15000,  # num_boost_round
        'subsample': 0.66,
        # min_split_gain
        'min_data_in_leaf': 1,  # min_child_samples
        # min_child_weight
        'rsm': 1,
        'leaf_estimation_method': 'Gradient',  # Newton Gradient Exact
        'one_hot_max_size': 30,
        'fold_len_multiplier': 2,

        'depth': 7,
        'max_leaves': 31,  # num_leaves
        'bagging_temperature': 1,
        'random_strength': 1,
        'l2_leaf_reg': 2.452612631133676,  # reg_lambda
        # reg_alpha

        'border_count': 128,

        'random_seed': 42,
        'thread_count': -1,
        'use_best_model': True,
        'logging_level': 'Verbose',
    },
]

if __name__ == '__main__':
    paths = [reg_path, bcls_path, mcls_path]
    params = [reg_params, bcls_params, mcls_params]
    for path, params in zip(paths, params):
        json.dump(params, open(path, 'w'), indent=4)
