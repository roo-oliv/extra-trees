import numpy as np

from extra_trees.tree.tree import ExtraTreeClassifier
from extra_trees.tree.tree import ExtraTreeRegressor


def test_extra_tree_regressor(circles):
    X, y = circles
    indices = np.random.permutation(len(X.data))
    X_train = X[indices[:-10]]
    y_train = y[indices[:-10]]
    X_test = X[indices[-10:]]
    y_test = y[indices[-10:]]

    regressor = ExtraTreeRegressor()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    assert len(predictions) == len(y_test)


def test_extra_tree_classifier(circles):
    X, y = circles
    indices = np.random.permutation(len(X.data))
    X_train = X[indices[:-10]]
    y_train = y[indices[:-10]]
    X_test = X[indices[-10:]]
    y_test = y[indices[-10:]]

    classifier = ExtraTreeClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    assert len(predictions) == len(y_test)

    all_classes = np.unique(y)
    predicted_classes = np.unique(predictions)
    assert all(value in all_classes for value in predicted_classes)
