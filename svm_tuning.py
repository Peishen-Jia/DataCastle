__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X, y = load_svmlight_file('./data/train.libsvm')
X = StandardScaler().fit_transform(X)
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
