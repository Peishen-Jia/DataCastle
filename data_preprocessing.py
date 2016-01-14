__author__ = 'basin'
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import dump_svmlight_file

train_x = pd.read_csv('./data/train_x.csv')
train_y = pd.read_csv('./data/train_y.csv')
features_type = pd.read_csv('./data/features_type.csv')
test_X = pd.read_csv('./data/test_x.csv')
train_unlabeled = pd.read_csv('./data/train_unlabeled.csv')

train_xy = pd.merge(train_x, train_y, on='uid')
y = train_xy['y']
X = train_xy.drop(['uid', 'y'], axis=1)
test_X = test_X.drop(['uid'], axis=1)
train_unlabeled = train_unlabeled.drop(['uid'], axis=1)

category_type = features_type[features_type.type == 'category']['feature'].values
numeric_type = features_type[features_type.type == 'numeric']['feature'].values

# one-hot-encoder requires the category value >= 0
# in order to scale the category value >= 0, you must get the GLOBAL min
full_data = train_unlabeled.append(X).append(test_X)
global_min_category = full_data[category_type].min()

full_data = pd.concat(
    [full_data[numeric_type], full_data[category_type].sub(global_min_category, axis=1)], axis=1)
full_data = full_data[sorted(full_data.columns.values)]

X = pd.concat([X[numeric_type], X[category_type].sub(global_min_category, axis=1)], axis=1)
X = X[sorted(X.columns.values)]

test_X = pd.concat(
    [test_X[numeric_type], test_X[category_type].sub(global_min_category, axis=1)], axis=1)
test_X = test_X[sorted(test_X.columns.values)]

train_unlabeled = pd.concat(
    [train_unlabeled[numeric_type],
     train_unlabeled[category_type].sub(global_min_category, axis=1)], axis=1)
train_unlabeled = train_unlabeled[sorted(train_unlabeled.columns.values)]

mask = pd.DataFrame(X.columns).isin(category_type)[0].values

one_hot_encoder = OneHotEncoder(categorical_features=mask)
# in order to be in agreement with X, text_X, train_unlabeled, must fit full data, then transform
one_hot_encoder.fit(full_data)

X = one_hot_encoder.transform(X)

test_X = one_hot_encoder.transform(test_X)

train_unlabeled = one_hot_encoder.transform(train_unlabeled)

dump_svmlight_file(X, y, './data/svmlight/train.libsvm', zero_based=True)

test_y = np.zeros(test_X.shape[0])

dump_svmlight_file(test_X, test_y, './data/svmlight/test.libsvm', zero_based=True)

train_unlabeled_y = np.zeros(train_unlabeled.shape[0])

dump_svmlight_file(train_unlabeled, train_unlabeled_y,
                   './data/svmlight/train_unlabeled.libsvm', zero_based=True)