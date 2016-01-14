__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np


def param_select(X, y):
    tuned_parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    print("# Tuning hyper-parameters for roc_auc with svc")
    clf = GridSearchCV(SVC(probability=True, class_weight='balanced', random_state=0),
                       tuned_parameters, cv=5, scoring='roc_auc', n_jobs=10)
    clf.fit(X, y)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))


def tuning(X, y):
    clf = SVC(C=1.0, kernel='rbf', gamma=0.001, class_weight='balanced', verbose=True,
              random_state=0)
    skf = StratifiedShuffleSplit(y, n_iter=2, test_size=0.2, random_state=0)
    fold = 1
    for train_index, val_index in skf:
        print "fold:", fold
        fold += 1
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf.fit(X_train, y_train)
        val_0_1 = pd.DataFrame(clf.predict_proba(X_val),
                               columns=["predict_0", "predict_1"])
        print '\tauc', roc_auc_score(y_val, val_0_1.predict_1.values)


if __name__ == "__main__":
    X, y = load_svmlight_file('./data/train.libsvm')
    X = StandardScaler().fit_transform(X.todense())
    # test_X, fake_y = load_svmlight_file('./data/test.libsvm')
    # test_uid = pd.read_csv('./data/test_x.csv')['uid']
    # gen_svc(X, y, test_X, test_uid)
    tuning(X, y)
