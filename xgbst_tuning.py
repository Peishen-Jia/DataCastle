__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

import time
import xgboost as xgb
import pandas as pd
import os


def param_select(X, y):
    tuned_parameters = {'reg_alpha': [1, 5, 10, 20, 50, 100],
                        'scale_pos_weight': [1.0, 1542.0 / 13458.0, 800.0 / 13458.0],
                        'max_depth': [6, 8, 10], 'n_estimators': [2000, 5000]}
    print("# Tuning hyper-parameters for roc_auc with xgboost")
    clf = GridSearchCV(xgb.XGBClassifier(learning_rate=0.1, objective='binary:logistic'),
                       tuned_parameters, cv=5, scoring='roc_auc', n_jobs=10)
    clf.fit(X, y)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))


def gen_xgboost(X, y, test_X, test_uid):
    clf = xgb.XGBClassifier(n_estimators=20000, scale_pos_weight=1.0, max_depth=6,
                            objective='binary:logistic', learning_rate=0.02,
                            gamma=0.1, min_child_weight=3, max_delta_step=5,
                            subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1,
                            reg_alpha=10, reg_lambda=3000)
    params = clf.get_params()
    print 'params:', params
    dir_name = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}'.format(
        str(params['scale_pos_weight']), str(params['max_depth']),
        str(params['learning_rate']), str(params['gamma']), str(params['min_child_weight']),
        str(params['max_delta_step']), str(params['subsample']), str(params['colsample_bytree']),
        str(params['colsample_bylevel']), str(params['reg_alpha']), str(params['reg_lambda']))
    if not os.path.exists('./data/results/' + dir_name):
        os.mkdir('./data/results/' + dir_name)

    skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=0)

    # skf = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=0)
    fold = 1
    for train_index, val_index in skf:
        print "fold:", fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # eval_metric use the parameters in XGBoost doc
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='auc', early_stopping_rounds=1000, verbose=True)
        # predict probability
        test_0_1 = pd.DataFrame(clf.predict_proba(test_X, ntree_limit=clf.best_iteration),
                                columns=["predict_0", "predict_1"])
        test_result = pd.DataFrame(columns=["uid", "score"])
        test_result.uid = test_uid
        test_result.score = test_0_1.predict_1
        # remember to edit xgb.csv , add ""
        test_result.to_csv("./data/results/" + dir_name + "/res_xgb_" + str(fold) + ".csv",
                           index=None, encoding='utf-8')
        fold += 1


def tuning_1(X, y):
    clf = xgb.XGBClassifier(n_estimators=20000, scale_pos_weight=1.0, max_depth=6,
                            objective='binary:logistic', learning_rate=0.02,
                            gamma=0.5, min_child_weight=3, max_delta_step=5,
                            subsample=0.8, colsample_bytree=0.4, colsample_bylevel=1,
                            reg_alpha=0, reg_lambda=3000, nthread=-1)
    print 'parameters:', clf.get_params()
    skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=0)
    for train_index, val_index in skf:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # eval_metric use the parameters in XGBoost doc
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='auc', early_stopping_rounds=1000, verbose=True)
        print "best_score", clf.best_score


if __name__ == "__main__":
    start = time.time()
    X, y = load_svmlight_file('./data/svmlight/train.libsvm')
    # test_X, fake_y = load_svmlight_file('./data/svmlight/test.libsvm')
    # test_uid = pd.read_csv('./data/test_x.csv')['uid']
    # gen_xgboost(X, y, test_X, test_uid)
    tuning_1(X, y)
    end = time.time()
    print 'run time(s):', (end - start)
