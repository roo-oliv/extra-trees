import numpy
from sklearn.utils import Bunch

from extra_trees.ensemble.tree import build_extra_tree
from sklearn import datasets


def test_build_extra_tree():
    numpy.random.seed(0)
    breast_cancer_dataset = datasets.load_breast_cancer()
    indices = numpy.random.permutation(len(breast_cancer_dataset.data))
    train = Bunch(
        data=breast_cancer_dataset.data[indices[:-10]],
        target=breast_cancer_dataset.target[indices[:-10]])
    test = Bunch(
        data=breast_cancer_dataset.data[indices[-10:]],
        target=breast_cancer_dataset.target[indices[-10:]])

    print("\nTRAINING")

    extra_tree = build_extra_tree(train, 2, 3)

    print("\nTESTING")

    predictions = [extra_tree(entry) for entry in test.data]

    print("\nRESULTS:")
    print("PREDICTIONS:", end=' ')
    print(predictions)
    print("ACTUAL VALUES:", end=' ')
    print(test.target)

    assert len(predictions) == len(test.target)  # TODO: real test
