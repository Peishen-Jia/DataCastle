__author__ = 'basin'
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import time
import xgboost as xgb


def tuning_xgbst(X, y):
    clf = xgb.XGBClassifier(n_estimators=10000,
                            scale_pos_weight=1.0, #1400.0/13458.0,
                            max_depth=6,
                            objective='binary:logistic',
                            learning_rate=0.02,
                            gamma=0.1,
                            min_child_weight=3,
                            max_delta_step=0,
                            subsample=0.7,
                            colsample_bytree=0.4,
                            colsample_bylevel=1.0,
                            reg_alpha=0,
                            reg_lambda=3000,
                            seed=0,
                            nthread=-1)
    print clf.get_params()
    # skf = StratifiedShuffleSplit(y, n_iter=5, test_size=0.25, random_state=0)
    skf = StratifiedKFold(y, n_folds=3, random_state=0)
    fold = 1
    for train_index, val_index in skf:
        print 'fold ', fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # eval_metric use the parameters in XGBoost doc
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='auc', early_stopping_rounds=1000, verbose=False)
        print "best_score", clf.best_score
        joblib.dump(clf, './models/xgbst/CV_' + str(fold) + '.model')
        fold += 1


def tuning_randomforest(X, y):
    clf = RandomForestClassifier(n_estimators=10000, criterion='entropy', max_depth=6,
                                 min_samples_split=2, min_samples_leaf=1,
                                 min_weight_fraction_leaf=0,
                                 max_features=0.2, n_jobs=-1, class_weight='balanced_subsample',
                                 verbose=0)
    print 'parameters:', clf.get_params()
    skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=0)
    for train_index, val_index in skf:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf.fit(X_train, y_train)
        print 'train accuracy', clf.score(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        print 'val auc:', roc_auc_score(y_val, y_val_pred)


if __name__ == "__main__":
    start = time.time()
    X, y = load_svmlight_file('./data/svmlight/train.libsvm')
    # test_X, fake_y = load_svmlight_file('./data/svmlight/test.libsvm')
    # test_uid = pd.read_csv('./data/test_x.csv')['uid']
    # gen_xgboost(X, y, test_X, test_uid)
    tuning_xgbst(X, y)
    # tuning_randomforest(X, y)
    end = time.time()
    print 'run time(s):', (end - start)
