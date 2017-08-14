from numbers import Number
from typing import Any

from sklearn.utils import Bunch

from extra_trees.ensemble.tree import build_extra_tree


class ExtraTreesModelBase:
    def __init__(
            self, n_estimators: int, n_features: int, min_size: int,
            numeric: bool = None):
        if n_features > min_size:
            raise ValueError("n_features cannot be greater than min_size")
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.min_size = min_size
        self.forest = list()
        if numeric is not None:
            self._type = Number if numeric else Any
        else:
            self._type = None

    def fit(self, train: Bunch):
        self.forest = [
            build_extra_tree(
                train, self.n_features, self.min_size, _type=self._type)
            for _ in range(self.n_estimators)
        ]

    def apply(self, dataset: Bunch):
        results = []
        for row in dataset.data:
            results.append(
                sum([tree(row) for tree in self.forest]) / len(self.forest))

        return results


class ExtraTreesRegressor(ExtraTreesModelBase):
    def __init__(self, n_estimators: int, n_features: int, min_size: int):
        super().__init__(n_estimators, n_features, min_size, True)


class ExtraTreesClassifier(ExtraTreesModelBase):
    def __init__(self, n_estimators: int, n_features: int, min_size: int):
        super().__init__(n_estimators, n_features, min_size, False)
