import re
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
#import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import random
from sklearn import preprocessing

raw_train = pd.read_csv("data/train/train_01.csv")
raw_test = pd.read_csv("data/test/test_01.csv")

merge = pd.concat([raw_train, raw_test])

#添加用户字段
user_feature_data = pd.read_csv("data/user_feature.csv",error_bad_lines=False)
user_feature_data = user_feature_data.drop(['user_index','top_genres_play_duration', 'top_tags_play_duration', 'top_genres', \
                                  'top_tags', 'rate_avg', 'rate_var_pop', 'rate_stddev_pop', 'min_rate', \
                                  'max_rate', 'middle_rating','regions_list','regin_duration_avg','regin_duration_var_pop','regin_duration_stddev_pop'], axis=1)

merge = pd.merge(merge, user_feature_data, how='left',left_on="user_id",right_on="user_id")

#添加物品字段
item_feature_data = pd.read_csv("data/item_feature.csv",error_bad_lines=False)
item_feature_data = item_feature_data.drop(['item_play_count1','item_play_count3','item_play_count7','item_ctr1','item_ctr3','item_ctr7'], axis=1)

merge = pd.merge(merge, item_feature_data, how='left',left_on="return_item_id", right_on="item_code")
merge = merge.drop(['item_code'], axis=1)

encoder = preprocessing.LabelEncoder()
encoder.fit(list(merge["user_id"].values))
merge["user_id"] = encoder.transform(list(merge["user_id"].values))
# encoder.fit(list(merge["user_id"].values))
# merge["user_id"] = encoder.transform(list(merge["user_id"].values))

encoder2 = preprocessing.LabelEncoder()
encoder2.fit(list(merge['return_item_id'].values))
merge['return_item_id'] = encoder2.transform(list(merge['return_item_id'].values))
# encoder2.fit(list(merge['return_item_id'].values))
# merge['return_item_id'] = encoder2.transform(list(merge['return_item_id'].values))

raw_train = merge[:len(raw_train)]
raw_test = merge[len(raw_train):]

# print(raw_train['play_ratio'])

# print(raw_train)

y_train_click = raw_train['is_click']
y_train_play = raw_train['is_play']
y_train_play_ratio = raw_train['play_ratio']
raw_train = raw_train.drop(['is_click', 'is_play', 'play_ratio'], axis=1)
X_train = raw_train.values

y_test_click = raw_test['is_click']
y_test_play = raw_test['is_play']
y_test_play_ratio = raw_test['play_ratio']
raw_test = raw_test.drop(['is_click', 'is_play', 'play_ratio'], axis=1)
X_test = raw_test.values

y_train = y_train_play
y_test = y_test_play
print(y_train.value_counts())
print(y_test.value_counts())

#XGBoost
'''
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train')]

params={'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 20,
        'lambda': 1,
        'subsample': 1,
        'colsample_bytree': 1,
        'min_child_weight': 1,
        'eta': 0.3,
        'seed': 0,
        'nthread': 8,
        'gamma': 0,
        'scale_pos_weight': 3,
        'learning_rate': 0.3}

bst = xgb.train(params, dtrain, num_boost_round = 20, evals=watchlist)
ypred = bst.predict(dtest)
print('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
'''

train_data = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)
test_data = lgb.Dataset(X_test, y_test, reference=train_data)

from hyperopt import fmin, tpe, hp, partial

# 自定义hyperopt的参数空间
space = {"max_depth": hp.randint("max_depth", 15),
         "num_trees": hp.randint("num_trees", 300),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "bagging_fraction": hp.randint("bagging_fraction", 5),
         "num_leaves": hp.randint("num_leaves", 6),
         }

def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['num_trees'] = argsDict['num_trees'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["bagging_fraction"] = argsDict["bagging_fraction"] * 0.1 + 0.5
    argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict

from sklearn.metrics import mean_squared_error

def lightgbm_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

    params = {'nthread': -1,  # 进程数
              'max_depth': argsDict['max_depth'],  # 最大深度
              'num_trees': argsDict['num_trees'],  # 树的数量
              'eta': argsDict['learning_rate'],  # 学习率
              'bagging_fraction': argsDict['bagging_fraction'],  # 采样数
              'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
              'objective': 'regression',
              'feature_fraction': 0.7,  # 样本列采样
              'lambda_l1': 0,  # L1 正则化
              'lambda_l2': 0,  # L2 正则化
              'bagging_seed': 100,  # 随机种子,light中默认为100
              }
    #params['metric'] = ['rmse']
    params['metric'] = ['auc']

    model_lgb = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data],early_stopping_rounds=1000)

    return get_tranformer_score(model_lgb)

def get_tranformer_score(tranformer):

    model = tranformer
    prediction = model.predict(X_test, num_iteration=model.best_iteration)

    #return mean_squared_error(y_test, prediction)
    return metrics.roc_auc_score(y_test, prediction)

# 开始使用hyperopt进行自动调参
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(lightgbm_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)


auc = lightgbm_factory(best)
# print('best :', best)
# print('best param after transform :')
# argsDict_tranform(best,isPrint=True)
# print('auc of the best lightgbm:', np.sqrt(auc))
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'max_depth': 3,
#     'objective': 'binary',
#     'metric': 'auc',
#     'min_child_samples': 10,
#     'num_leaves': 30,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'min_data_in_leaf': 50,
#     'min_sum_heassian_in_leaf': 50,
#     'bagging_fraction': 0.85,
#     'bagging_freq': 5,
#     'lambda_l1': 5,
#     'lambda_l2': 5,
#     'verbose': 0
# }
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=lgb_test)
#                 #early_stopping_rounds=5)
# ypred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#
# print('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
