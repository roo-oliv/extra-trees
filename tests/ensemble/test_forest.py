import numpy as np
from sklearn import datasets

from extra_trees.ensemble.forest import ExtraTreesRegressor


def test_extra_trees_regressor():
    boston_dataset = datasets.load_boston()
    indices = np.random.permutation(len(boston_dataset.data))
    X_train = boston_dataset.data[indices[:-10]]
    y_train = boston_dataset.target[indices[:-10]]
    X_test = boston_dataset.data[indices[-10:]]
    y_test = boston_dataset.target[indices[-10:]]

    regressor = ExtraTreesRegressor()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print('\n')
    print(predictions)
    print(y_test)

    assert len(predictions) == len(y_test)
