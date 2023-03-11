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

lgb_train = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'max_depth': 3,
    'objective': 'binary',
    'metric': 'auc',
    'min_child_samples': 10,
    'num_leaves': 30,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'min_data_in_leaf': 50,
    'min_sum_heassian_in_leaf': 50,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 5,
    'lambda_l2': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_test)
                #early_stopping_rounds=5)
ypred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

print('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
