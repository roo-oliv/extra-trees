from sklearn import datasets

import numpy
from sklearn.utils import Bunch

from extratrees.ensemble.forest import ExtraTreesRegressor, \
    ExtraTreesClassifier


def test_extra_tree_regressor():
    numpy.random.seed(0)
    breast_cancer_dataset = datasets.load_breast_cancer()
    indices = numpy.random.permutation(len(breast_cancer_dataset.data))
    train = Bunch(
        data=breast_cancer_dataset.data[indices[:-10]],
        target=breast_cancer_dataset.target[indices[:-10]])
    test = Bunch(
        data=breast_cancer_dataset.data[indices[-10:]],
        target=breast_cancer_dataset.target[indices[-10:]])
    model = ExtraTreesRegressor(10, 3, 4)

    print("\nTRAINING")

    model.fit(train)

    print("\nTESTING")

    predictions = model.apply(test)

    print("\nRESULTS:")
    print("PREDICTIONS:", end=' ')
    print(predictions)
    print("ACTUAL VALUES:", end=' ')
    print(test.target)

    assert len(predictions) == len(test.target)  # TODO: real test


def test_extra_tree_classifier():
    numpy.random.seed(0)
    breast_cancer_dataset = datasets.load_breast_cancer()
    indices = numpy.random.permutation(len(breast_cancer_dataset.data))
    train = Bunch(
        data=breast_cancer_dataset.data[indices[:-10]],
        target=breast_cancer_dataset.target[indices[:-10]])
    test = Bunch(
        data=breast_cancer_dataset.data[indices[-10:]],
        target=breast_cancer_dataset.target[indices[-10:]])
    model = ExtraTreesClassifier(10, 3, 4)

    print("\nTRAINING")

    model.fit(train)

    print("\nTESTING")

    predictions = model.apply(test)

    print("\nRESULTS:")
    print("PREDICTIONS:", end=' ')
    print(predictions)
    print("ACTUAL VALUES:", end=' ')
    print(test.target)

    assert len(predictions) == len(test.target)  # TODO: real test
