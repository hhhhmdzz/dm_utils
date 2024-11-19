import os
import json

# file directory
root = os.path.dirname(__file__)

reg_path = os.path.join(root, 'reg.json')
reg_params = [
    {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',

        'booster': 'gbtree',
        'eta': 0.03,  # lr
        # epoch
        'gamma': 0.1,  # min_split_gain
        # min_data_in_leaf
        'min_child_weight': 3,

        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'lambda': 10,
        'alpha': 0,

        'missing': -999.0,
        'seed': 42,
        'nthread': -1,
        'verbosity': 0,
    },
    {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 571,
        'verbosity': 1,
        'objective': 'reg:squarederror',
        'booster': 'dart',
        'n_jobs': -1,
        'gamma': 0,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'reg_alpha': 6.345561548486771,
        'reg_lambda': 2.808394786832976,
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'importance_type': 'gain',
        'num_round': 223,
    },
    {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 749,
        'verbosity': 1,
        'objective': 'reg:squarederror',
        'booster': 'dart',
        'n_jobs': -1,
        'gamma': 0,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'reg_alpha': 7.819025434063891,
        'reg_lambda': 0.005996946163704,
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'importance_type': 'gain',
        'num_round': 934,
    },
    {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 731,
        'verbosity': 1,
        'objective': 'reg:squarederror',
        'booster': 'dart',
        'n_jobs': -1,
        'gamma': 0,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'reg_alpha': 9.299861941127418,
        'reg_lambda': 8.762447153395918,
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'importance_type': 'gain',
        'num_round': 917,
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
    {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'booster': 'gbtree',
        'eta': 0.03,  # lr
        # epoch
        'gamma': 0.1,  # min_split_gain
        # min_data_in_leaf
        'min_child_weight': 3,

        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'lambda': 10,
        'alpha': 0,

        'missing': -999.0,
        'seed': 42,
        'nthread': -1,
        'verbosity': 0,
    }
]

mcls_path = os.path.join(root, 'mcls.json')
mcls_params = [
    {
        'objective': 'multi:softprob',
        # 'num_class': 3,
        'eval_metric': 'auc',

        'booster': 'gbtree',
        'eta': 0.03,  # lr
        # epoch
        'gamma': 0.1,  # min_split_gain
        # min_data_in_leaf
        'min_child_weight': 3,

        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'lambda': 10,
        'alpha': 0,

        'missing': -999.0,
        'seed': 42,
        'nthread': -1,
        'verbosity': 0,
    },
    {
    'objective': 'multi:softmax',
        # 'num_class': 3,
        'eval_metric': 'auc',

        'booster': 'gbtree',
        'eta': 0.1,  # lr
        # epoch
        'gamma': 0.1,  # min_split_gain
        # min_data_in_leaf
        'min_child_weight': 3,

        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'lambda': 2,
        'alpha': 0,

        'missing': -999.0,
        'seed': 42,
        'nthread': -1,
        'verbosity': 0,
    }
]

if __name__ == '__main__':
    paths = [reg_path, bcls_path, mcls_path]
    params = [reg_params, bcls_params, mcls_params]
    for path, params in zip(paths, params):
        json.dump(params, open(path, 'w'), indent=4)
