__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.externals import joblib

import time
import xgboost as xgb
import pandas as pd
import os


def get_params(dir_name, file_name):
    """
    read the parameters from a file
    :param file_name:
    :return: a dictionary containing the parameters
    """
    param_str = open(dir_name + '/' + file_name).read()
    return eval(param_str)


def gen_xgboost(X, y, test_X, test_uid, dir_name, file_name):
    params = get_params(dir_name, file_name)
    clf = xgb.XGBClassifier(n_estimators=20000,
                            scale_pos_weight=params['scale_pos_weight'],
                            max_depth=params['max_depth'],
                            objective='binary:logistic', learning_rate=0.02,
                            gamma=params['gamma'], min_child_weight=params['min_child_weight'],
                            max_delta_step=params['max_delta_step'], subsample=params['subsample'],
                            colsample_bytree=params['colsample_bytree'],
                            colsample_bylevel=params['colsample_bylevel'],
                            reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'])

    skf = StratifiedShuffleSplit(y, n_iter=5, test_size=0.25, random_state=0)

    fold = 1
    for train_index, val_index in skf:
        if fold != 2:
            fold += 1
            continue

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
        test_result.to_csv("./models/xgbst/" + file_name + ".csv", index=None, encoding='utf-8')
        # save the model to disk
        joblib.dump(clf, './models/xgbst/' + file_name + '.model')
        fold += 1


if __name__ == "__main__":
    X, y = load_svmlight_file('./data/svmlight/train.libsvm')
    test_X, fake_y = load_svmlight_file('./data/svmlight/test.libsvm')
    test_uid = pd.read_csv('./data/test_x.csv')['uid']
    dir_name = './models/xgbst'
    files = os.listdir(dir_name)


    def f(x): return x.endswith('.json')


    files = filter(f, files)
    for file_name in files:
        start = time.time()
        gen_xgboost(X, y, test_X, test_uid, dir_name, file_name)
        end = time.time()
        print file_name + 'run time(s):', (end - start)
