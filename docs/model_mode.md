# **model_mode**

`model_mode` 由 `mode1` 和 `mode2` 两个元素组成，可以表示不同模型的不同接口，不同模型不同接口在分类/回归、训练/预测、数据类型等方面有不同的支持，因此设计了模型模式（简称模式） `model_mode` 方面后续处理

| `mode1`  | `mode2`  |            说明             |
| :------: | :------: | :-------------------------: |
| sklearn  | sklearn  |  sklearn 模型 sklearn 接口  |
| sklearn  | xgboost  |  xgboost 模型 sklearn 接口  |
| sklearn  | lightgbm | lightgbm 模型 sklearn 接口  |
| sklearn  | catboost | catboost 模型 sklearn 接口  |
| sklearn  | ngboost  |  ngboost 模型 sklearn 接口  |
| sklearn  |  tabnet  |  tabnet 模型 sklearn 接口   |
| xgboost  | xgboost  |  xgboost 模型 xgboost 接口  |
| lightgbm | lightgbm | lightgbm 模型 lightgbm 接口 |
| catboost | catboost | catboost 模型 catboost 接口 |

不同模型的继承关系如下（以分类为例，回归同理）：

|   模型   |                            类                             |        说明        |
| :------: | :-------------------------------------------------------: | :----------------: |
| sklearn  |                      `BaseEstimator`                      |                    |
| xgboost  |  `XGBClassifier(XGBModel(BaseEstimator))`、`xgb.Booster`  |   独立的原生接口   |
| lightgbm | `LGBMClassifier(LGBMModel(BaseEstimator))`、`lgb.Booster` |   独立的原生接口   |
| catboost |              `CatBoostClassifier(CatBoost)`               | 支持 sklearn  接口 |
| ngboost  |          `NGBClassifier(NGBoost, BaseEstimator)`          |                    |
|  tabnet  |                 `TabModel(BaseEstimator)`                 |                    |

不同模式在不同任务（分类/回归）不同阶段（训练/预测）下的数据（特征/标签）格式如下：

|  `mode1`, `mode2`  | 训练、预测输入数据类型 | 分类、回归训练输入标签形状 |   分类预测输出标签、概率形状    | 回归预测输出形状 |
| :----------------: | :--------------------: | :------------------------: | :-----------------------------: | :--------------: |
|  sklearn, sklearn  |         np, pd         |       (n, ), (n, 1)        | (n, )、(n, 2), (n, num_classes) |      (n, )       |
|  sklearn, xgboost  |         np, pd         |       (n, ), (n, 1)        | (n, )、(n, 2), (n, num_classes) |      (n, )       |
| sklearn, lightgbm  |         np, pd         |       (n, ), (n, 1)        | (n, )、(n, 2), (n, num_classes) |      (n, )       |
| sklearn, catboost  |         np, pd         |       (n, ), (n, 1)        | (n, )、(n, 2), (n, num_classes) |      (n, )       |
|  sklearn, ngboost  |         np, pd         |       (n, ), (n, 1)        | (n, )、(n, 2), (n, num_classes) |      (n, )       |
|  sklearn, tabnet   |           np           |       (n, ), (n, 1)        | (n, )、(n, 2), (n, num_classes) |      (n, 1)      |
|  xgboost, xgboost  |      xgb.DMatrix       |       (n, ), (n, 1)        | (n, )、(n, ), (n, num_classes)  |      (n, )       |
| lightgbm, lightgbm |  lgb.Dataset、np, pd   |       (n, ), (n, 1)        | (n, )、(n, ), (n, num_classes)  |      (n, )       |
| catboost, catboost |        cb.Pool         |       (n, ), (n, 1)        | (n, )、(n, ), (n, num_classes)  |      (n, )       |

对于某些存在特征重要性的模型（比如树模型、TabNet 等），其获取特征名字和特征重要性的接口也不同，因此需要根据 `model_mode` 来选择不同的接口，具体如下：

|  `mode1`, `mode2`  |          特征名字         |       特征重要性       |      说明       |
| :----------------: | :----------------------------------: | :-------------: | :----------------: |
|  sklearn, sklearn  | `model.feature_names_in_` | `model.feature_importances_` |               |
|  sklearn, xgboost  | `model.feature_names_in_` | `model.feature_importances_` |               |
| sklearn, lightgbm  | `model.feature_name_` | `model.feature_importances_` |               |
| sklearn, catboost | `model.feature_names_` | `model.feature_importances_` |  |
| sklearn, ngboost |  | `model.feature_importances_` | (loc trees, scale trees) |
| sklearn, tabnet |  | `model.feature_importances_` |  |
| xgboost, xgboost | `model.get_score(importance_type).keys()` | `model.get_score(importance_type).values()` |  |
| lightgbm, lightgbm | `model.feature_name()` | `model.feature_importance(importance_type)` |  |
| catboost, catboost | `model.feature_names_` | `model.get_feature_importance()` |  |

参数所在类/方法：

|  `mode1`, `mode2`  |      `epochs` 训练轮数      |   `eval_rounds` 验证轮数    |     `early_stop_rounds` 早停轮数     |      `log_level` 日志等级      |
| :----------------: | :-------------------------: | :-------------------------: | :----------------------------------: | :----------------------------: |
|  sklearn, sklearn  | BaseEstimator(n_estimators) |              -              |                  -                   |               -                |
|  sklearn, xgboost  | BaseEstimator(n_estimators) |        fit(verbose)         | BaseEstimator(early_stopping_rounds) |    BaseEstimator(verbosity)    |
| sklearn, lightgbm  | BaseEstimator(n_estimators) |       fit(callbacks)        |            fit(callbacks)            |    BaseEstimator(verbosity)    |
| sklearn, catboost  |  BaseEstimator(iterations)  |      fit(verbose_eval)      |      fit(early_stopping_rounds)      |  BaseEstimator(logging_level)  |
|  sklearn, ngboost  | BaseEstimator(n_estimators) | BaseEstimator(verbose_eval) | BaseEstimator(early_stopping_rounds) |               -                |
|  sklearn, tabnet   |       fit(max_epochs)       |   BaseEstimator(verbose)    |            fit(patience)             |               -                |
|  xgboost, xgboost  |   train(num_boost_round)    |     train(verbose_eval)     |     train(early_stopping_rounds)     |   train(params['verbosity'])   |
| lightgbm, lightgbm |   train(num_boost_round)    |      train(callbacks)       |           train(callbacks)           |   train(params['verbosity'])   |
| catboost, catboost |      train(iterations)      |     train(verbose_eval)     |     train(early_stopping_rounds)     | train(params['logging_level']) |

|  `mode1`, `mode2`  | `data` 训练数据  | `data_evals` 验证数据 |      |      |
| :----------------: | :--------------: | :-------------------: | :--: | :--: |
|  sklearn, sklearn  |    fit(x, y)     |           -           |      |      |
|  sklearn, xgboost  |    fit(x, y)     |     fit(eval_set)     |      |      |
| sklearn, lightgbm  |    fit(x, y)     |     fit(eval_set)     |      |      |
| sklearn, catboost  |    fit(x, y)     |     fit(eval_set)     |      |      |
|  sklearn, ngboost  |    fit(x, y)     |   fit(X_val, Y_val)   |      |      |
|  sklearn, tabnet   |    fit(x, y)     |     fit(eval_set)     |      |      |
|  xgboost, xgboost  |  train(dtrain)   |     train(evals)      |      |      |
| lightgbm, lightgbm | train(train_set) |   train(valid_sets)   |      |      |
| catboost, catboost |  train(dtrain)   |    train(eval_set)    |      |      |

>注：
>ngboost 只有 `NGBRegressor` 支持 `early_stop_rounds`，`NGBClassifier` 不支持

特征、特诊重要性属性/方法：

|  `mode1`, `mode2`  |   `feature` 特征   | `importance` 特征重要性 |      |
| :----------------: | :----------------: | :---------------------: | :--: |
|  sklearn, sklearn  | feature_names_in_  |  feature_importances_   |      |
|  sklearn, xgboost  | feature_names_in_  |  feature_importances_   |      |
| sklearn, lightgbm  |   feature_name_    |  feature_importances_   |      |
| sklearn, catboost  |   feature_names_   |  feature_importances_   |      |
|  sklearn, ngboost  |         -          |  feature_importances_   |      |
|  sklearn, tabnet   |         -          |  feature_importances_   |      |
|  xgboost, xgboost  | get_score().keys() |  get_score().values()   |      |
| lightgbm, lightgbm |   feature_name()   |  feature_importance()   |      |
| catboost, catboost |   feature_names_   |  feature_importances_   |      |
