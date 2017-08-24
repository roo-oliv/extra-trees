"""
Authors: Rodrigo Martins de Oliveira
         Felipe Anchieta Santos Costa
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor as SKExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

from extra_trees.ensemble.forest import ExtraTreesRegressor

# Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
y = array[:, 8]

# Use 6-fold cross-validation
cv = StratifiedKFold(n_splits=6)

# Models
models = [
    ('SVR', SVR()),
    ('AdaBoost', AdaBoostRegressor()),
    ('RandomForest', RandomForestRegressor(min_samples_split=5)),
    ('ExtraTrees (SciKit)', SKExtraTreesRegressor(min_samples_split=5)),
    ('ExtraTrees', ExtraTreesRegressor(min_samples_split=5)),
]

folds = list(cv.split(X, y))

# Train & Test regression models
names = []
explained_variance_scores = []
mean_absolute_errors = []
mean_squared_errors = []
mean_squared_log_errors = []
median_absolute_errors = []
r2_scores = []
for name, model in models:
    names.append(name)
    evs = mae = mse = msle = mdae = r2s = 0
    for train, test in folds:
        predictions = model.fit(X[train], y[train]).predict(X[test])
        evs += explained_variance_score(y[test], predictions)
        mae += mean_absolute_error(y[test], predictions)
        mse += mean_squared_error(y[test], predictions)
        msle += mean_squared_log_error(y[test], predictions)
        mdae += median_absolute_error(y[test], predictions)
        r2s += r2_score(y[test], predictions)

    evs /= len(folds)
    mae /= len(folds)
    mse /= len(folds)
    msle /= len(folds)
    mdae /= len(folds)
    r2s /= len(folds)

    explained_variance_scores.append(evs)
    mean_absolute_errors.append(mae)
    mean_squared_errors.append(mse)
    mean_squared_log_errors.append(msle)
    median_absolute_errors.append(mdae)
    r2_scores.append(r2s)

    print(
        '{0} & {1:0.3f} & {2:0.3f} & {3:0.3f}'
        ' & {4:0.3f} & {5:0.3f} & {6:0.3f} \\\\ \hline'
        .format(name, evs, mae, mse, msle, mdae, r2s))
