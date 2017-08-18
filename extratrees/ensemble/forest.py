from collections import Counter
from numbers import Number
from typing import Any

import numpy
from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import Bunch

from extratrees.ensemble.tree import build_extra_tree


class ExtraTreesModelBase:
    def __init__(
            self, n_estimators: int, min_size: int, n_features: int = None,
            numeric: bool = None, n_jobs: int = 1):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.min_size = min_size
        self.forest = list()
        self.n_jobs = n_jobs
        self.class_weight = None
        if numeric is not None:
            self._type = Number if numeric else Any
        else:
            self._type = None

    @property
    def estimators_(self):
        return self.forest

    def fit(self, x, y, sample_weight=None):
        self.forest = [
            build_extra_tree(
                x, y, self.min_size, self.n_features, _type=self._type)
            for _ in range(self.n_estimators)
        ]

        return self

    def apply(self, dataset: Bunch):
        results = []
        for row in dataset.data:
            predictions = [tree(row) for tree in self.forest]
            if self._type == Number:
                results.append(sum(predictions) / len(predictions))
            else:
                s = Counter({})
                for prediction in predictions:
                    s += prediction
                results.append(s)

        return results

    def predict(self, x):
        return numpy.asarray(
            [max(p, key=p.get) for p in self.apply(Bunch(data=x))],
            dtype=numpy.int)


class ExtraTreesRegressor(ExtraTreesModelBase, ForestRegressor):
    def __init__(
            self, n_estimators: int, min_size: int, n_features: int = None):
        super().__init__(n_estimators, min_size, n_features, True)


class ExtraTreesClassifier(ExtraTreesModelBase, ForestClassifier):
    def __init__(
            self, n_estimators: int, min_size: int, n_features: int = None):
        super().__init__(n_estimators, min_size, n_features, False)

    def fit(self, x, y, sample_weight=None):
        self.n_outputs_ = len(y)
        y, expanded_class_weight = self._validate_y_class_weight(y)
        return super().fit(x, y, sample_weight)
