import os
import json

# file directory
root = os.path.dirname(__file__)

reg_path = os.path.join(root, 'reg.json')
reg_params = [
    {
        'objective': 'regression',
        'metric': 'mae,mse',
        'eval_metric': 'mae,mse',

        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_boost_round': 100,  # n_estimators
        'min_split_gain': 0,
        'min_child_samples': 20,  # min_data_in_leaf
        'min_child_weight': 1e-3,  # min_sum_hessian_in_leaf

        'max_depth': -1,
        'num_leaves': 63,
        'bagging_fraction': 0.8,
        'bagging_fraction_seed': 42,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 42,
        'reg_lambda': 5,  # lambda_l2
        'reg_alpha': 2,  # lambda_l1

        'num_threads': -1,
        'verbose': -1,
    },
    {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'min_child_weight': 5,
        'num_leaves': 2 ** 7,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'learning_rate': 0.1,
        'seed': 2023,
        'nthread': 16,
        'verbose': -1,
        # 'device':'gpu'
    },
]

bcls_path = os.path.join(root, 'bcls.json')
bcls_params = [
    {
        'objective': 'binary',
        'metric': 'auc',
        'eval_metric': 'auc',

        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_boost_round': 100,  # n_estimators
        'min_split_gain': 0,
        'min_child_samples': 20,  # min_data_in_leaf
        'min_child_weight': 1e-3,  # min_sum_hessian_in_leaf

        'max_depth': 6,
        'num_leaves': 63,
        'bagging_fraction': 0.8,
        'bagging_fraction_seed': 42,
        'bagging_freq': 5,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 42,
        'reg_lambda': 5,  # lambda_l2
        'reg_alpha': 1.5,  # lambda_l1

        'num_threads': -1,
        'verbose': -1,
    },
]

mcls_path = os.path.join(root, 'mcls.json')
mcls_params = [
    {
        'objective': 'multiclass',
        'num_class': None,
        'metric': 'multi_logloss,multi_error',
        'eval_metric': 'multi_logloss,multi_error',

        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_boost_round': 100,  # n_estimators
        'min_split_gain': 0,
        'min_child_samples': 4,  # min_data_in_leaf
        'min_child_weight': 1e-3,  # min_sum_hessian_in_leaf

        'max_depth': 6,
        'num_leaves': 63,
        'bagging_fraction': 0.8,
        'bagging_fraction_seed': 42,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 42,
        'reg_lambda': 5,  # lambda_l2
        'reg_alpha': 1.5,  # lambda_l1

        'num_threads': -1,
        'verbose': -1,
    },
]

if __name__ == '__main__':
    paths = [reg_path, bcls_path, mcls_path]
    params = [reg_params, bcls_params, mcls_params]
    for path, params in zip(paths, params):
        json.dump(params, open(path, 'w'), indent=4)
