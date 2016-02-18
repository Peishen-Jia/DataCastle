__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from timeit import Timer
import numpy as np
import matplotlib.pyplot as plt


def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)


def logistic_regression():
    X, y = load_svmlight_file('./data/svmlight/train.libsvm')
    clf = LogisticRegression(penalty='l2', class_weight='auto', C=10)
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.2, random_state=0)
    for train_index, val_index in sss:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf.fit(X_train, y_train)
        y_val_lr = clf.predict_proba(X_val)[:, 1]
        y_val_lr = [1 if prab > 0.5 else 0 for prab in y_val_lr]
        print 'ROC AUC Score:', roc_auc_score(y_val, y_val_lr)

if __name__ == "__main__":
    t = Timer("logistic_regression()", setup="from __main__ import logistic_regression")
    print 'run time:', t.timeit(number=1)