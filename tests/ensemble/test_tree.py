import numpy
from sklearn.utils import Bunch

from extra_trees.ensemble.tree import build_extra_tree
from sklearn import datasets


def test_build_extra_tree():
    numpy.random.seed(0)
    iris_dataset = datasets.load_iris()
    indices = numpy.random.permutation(len(iris_dataset.data))
    train = Bunch(data=iris_dataset.data[indices[:-10]], target=iris_dataset.target[indices[:-10]])
    test = Bunch(data=iris_dataset.data[indices[-10:]], target=iris_dataset.target[indices[-10:]])
    extra_tree = build_extra_tree(train, 2, 3)

    predictions = [extra_tree(entry) for entry in test]

    assert len(predictions) == len(test)  # TODO: real test
