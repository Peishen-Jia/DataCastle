__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit

import xgboost as xgb
import pandas as pd


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
                            reg_alpha=0, reg_lambda=3000)
    skf = StratifiedShuffleSplit(y, n_iter=2, test_size=0.2, random_state=0)

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
        test_result.to_csv("./data/results/res_xgb_" + str(fold) + ".csv", index=None,
                           encoding='utf-8')
        fold += 1


def tuning(X, y):
    clf = xgb.XGBClassifier(n_estimators=20000, scale_pos_weight=1.0, max_depth=8,
                            objective='binary:logistic', learning_rate=0.02,
                            gamma=0.1, min_child_weight=3, max_delta_step=5,
                            subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1,
                            reg_alpha=0, reg_lambda=3000)
    skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=0)
    fold = 1
    for train_index, val_index in skf:
        print "fold:", fold
        fold += 1
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # eval_metric use the parameters in XGBoost doc
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='auc', early_stopping_rounds=1000, verbose=True)


if __name__ == "__main__":
    X, y = load_svmlight_file('./data/train.libsvm')
    test_X, fake_y = load_svmlight_file('./data/test.libsvm')
    test_uid = pd.read_csv('./data/test_x.csv')['uid']
    gen_xgboost(X, y, test_X, test_uid)
    # tuning(X, y)
