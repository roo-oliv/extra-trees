# Modified for MCZA015-13 class project by Rodrigo Martins de Oliveira
# License: BSD Style.

import matplotlib.pyplot as plt
import pandas
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from extra_trees.ensemble.forest import ExtraTreesClassifier

# prepare models
classification_models = [
    ('Logistic', LogisticRegression()),
    ('Nearest Neighbors', KNeighborsClassifier()),
    ('SVM', SVC()),
    ('DecisionTree', DecisionTreeClassifier()),
    ('RandomForest', RandomForestClassifier()),
    ('ExtraTrees (SciKit)', SKExtraTreesClassifier()),
    ('ExtraTrees', ExtraTreesClassifier()),
]
seed = 7

print("breast_cancer")
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in classification_models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('breast_cancer')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print("iris")
iris = load_iris()
X, y = iris.data, iris.target
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in classification_models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('iris')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print("wine")
wine = load_wine()
X, y = wine.data, wine.target
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in classification_models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('wine')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print("diabetes")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in classification_models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
