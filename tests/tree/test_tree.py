import numpy as np
from sklearn import datasets

from extra_trees.tree.tree import ExtraTreeRegressor


def test_extra_tree_regressor():
    boston_dataset = datasets.load_boston()
    indices = np.random.permutation(len(boston_dataset.data))
    X_train = boston_dataset.data[indices[:-10]]
    y_train = boston_dataset.target[indices[:-10]]
    X_test = boston_dataset.data[indices[-10:]]
    y_test = boston_dataset.target[indices[-10:]]

    regressor = ExtraTreeRegressor()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print('\n')
    print(predictions)
    print(y_test)
