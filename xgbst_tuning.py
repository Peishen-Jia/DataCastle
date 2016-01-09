__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

X, y = load_svmlight_file('./data/train.libsvm')
tuned_parameters = {'reg_alpha': [1, 5, 10, 20, 50, 100],
                    'scale_pos_weight': [1.0, 1542.0 / 13458.0, 800.0 / 13458.0],
                    'max_depth': [4, 6, 8, 10], 'n_estimators': [500, 1000, 2000, 5000]}
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
