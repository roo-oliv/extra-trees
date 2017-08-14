from sklearn.utils import Bunch

from extra_trees.ensemble.tree import build_extra_tree


class ExtraTreesRegressor:
    def __init__(self, n_estimators: int, n_features: int, min_size: int):
        if n_features > min_size:
            raise ValueError("n_features cannot be greater than min_size")
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.min_size = min_size
        self.forest = list()

    def fit(self, train: Bunch):
        self.forest = [
            build_extra_tree(train, self.n_features, self.min_size)
            for _ in range(self.n_estimators)
        ]

    def apply(self, dataset: Bunch):
        results = []
        for row in dataset.data:
            results.append(
                sum([tree(row) for tree in self.forest]) / len(self.forest))

        return results
