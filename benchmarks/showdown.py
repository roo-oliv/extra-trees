import numpy
import pandas
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from extratrees.ensemble.forest import ExtraTreesClassifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    ExtraTreesClassifier(n_estimators=10, min_size=1)]

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]

breast_cancer = load_breast_cancer()
X_folds = numpy.array_split(breast_cancer.data, 3)
y_folds = numpy.array_split(breast_cancer.target, 3)

k_fold = KFold()


for clf in classifiers:
    name = clf.__class__.__name__
    print("=" * 30)
    print(name)

    acc = 0.0
    ll = 0.0
    divisor = 0
    for train, test in k_fold.split(breast_cancer.data):
        divisor += 1

        clf.fit(breast_cancer.data[train], breast_cancer.target[train])

        train_predictions = clf.predict(breast_cancer.data[test])
        acc += accuracy_score(breast_cancer.target[test], train_predictions)

        train_predictions = clf.predict_proba(breast_cancer.data[test])
        ll += log_loss(breast_cancer.target[test], train_predictions)

    print("Accuracy: {:.4%}".format(acc/divisor))
    print("Log Loss: {}".format(ll/divisor))

    print('****Results****')

print("=" * 30)
