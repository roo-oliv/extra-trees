import numpy as np

from extra_trees.ensemble.forest import ExtraTreesClassifier
from extra_trees.ensemble.forest import ExtraTreesRegressor


def test_extra_trees_regressor(circles):
    X, y = circles
    indices = np.random.permutation(len(X.data))
    X_train = X[indices[:-10]]
    y_train = y[indices[:-10]]
    X_test = X[indices[-10:]]
    y_test = y[indices[-10:]]

    regressor = ExtraTreesRegressor()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    assert len(predictions) == len(y_test)


def test_extra_trees_classifier(circles):
    X, y = circles
    indices = np.random.permutation(len(X.data))
    X_train = X[indices[:-10]]
    y_train = y[indices[:-10]]
    X_test = X[indices[-10:]]
    y_test = y[indices[-10:]]

    classifier = ExtraTreesClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    assert len(predictions) == len(y_test)

    all_classes = np.unique(y)
    predicted_classes = np.unique(predictions)
    assert all(value in all_classes for value in predicted_classes)
