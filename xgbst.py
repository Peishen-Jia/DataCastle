__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

import xgboost as xgb

X, y = load_svmlight_file('./data/train.libsvm')
tuned_parameters = {'reg_alpha':[1, 10, 100, 200, 500],
                     'scale_pos_weight':[1.0, 1542.0/13458.0]}
scores = ['accuracy', 'roc_auc']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=8,
                                         objective='binary:logistic'), tuned_parameters,
                       cv=5, scoring='%s' % score, n_jobs=10)
    clf.fit(X, y)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

# xgbst = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=10, gamma=0.01,
#                           silent=False, scale_pos_weight=1542.0/13458.0, reg_alpha=1, reg_lambda=0)
# print cross_val_score(xgbst, X, y, scoring='roc_auc', cv=3, n_jobs=4)
# print xgbst.fit(X, y, eval_metric='auc', early_stopping_rounds=100)

