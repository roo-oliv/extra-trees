import random
from numbers import Number
from typing import Union, Callable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
from sklearn.tree._tree import issparse
from sklearn.tree.tree import DOUBLE
from sklearn.tree.tree import DTYPE
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class ExtraTree:
    def __init__(
            self, max_features: int, min_samples_split: int, n_classes_: int,
            n_outputs_: int, classification: bool = False):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_classes_ = n_classes_
        self.n_outputs_ = n_outputs_
        self.classification = classification
        self.decision_tree = ExtraTree.DecisionTree()

    class Leaf:
        def __init__(self, prediction):
            self.prediction = prediction

        def predict(self, _):
            return self.prediction

    class Node:
        def __init__(self, split: Callable[[object], bool]):
            self.split = split
            self.left = ExtraTree.DecisionTree()
            self.right = ExtraTree.DecisionTree()

        def predict(self, x):
            if self.split(x):
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    class DecisionTree:
        def __init__(self):
            self.node = None  # type: Union['ExtraTree.Node', 'ExtraTree.Leaf']

        def assign(self, node: Union['ExtraTree.Node', 'ExtraTree.Leaf']):
            self.node = node

        def predict(self, x):
            return self.node.predict(x)

    def _stop(self, X: np.ndarray, y: np.ndarray, constant: list):
        """
        :returns: `True` if stop condition to keep on branching is
                  reach, `False` otherwise
        """
        # Check if are at least `min_samples_split` in the data set
        if X.shape[1] < self.min_samples_split:
            return True

        # Check if there is at least one non-constant attribute in the
        # data set
        if not False in constant:
            return True

        # Check if the target attribute is non-constant
        if all(e == y[0][0] for e in y[:, 0]):
            return True

        return False

    @staticmethod
    def _pick_random_split(
            samples: list, attribute_index: int) -> Callable[[object], bool]:
        """
        Randomly chooses a pivot point (for numerical attributes) or
        a subset (for categorical attributes) and returns a test function
        which splits a tree branch into two other branches based on the
        chosen pivot or subset.
        """
        if isinstance(samples[0], Number):
            pivot = random.uniform(min(samples), max(samples))

            def test(x: np.array) -> bool:
                return x[attribute_index] < pivot

            test.type = 'numerical'
            test.attribute = attribute_index
            test.criteria = pivot
        else:
            values = set(samples)
            subset = random.sample(values, random.randint(1, len(values)-1))

            def test(x: np.array) -> bool:
                return x[attribute_index] in subset

            test.type = 'categorical'
            test.attribute = attribute_index
            test.criteria = subset

        return test

    @staticmethod
    def _split_sample(
            split: Callable[[object], bool], X: np.ndarray, y: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split X, y sample set in two with a split function
        :return: ((X_left, y_left), (X_right, y_right))
        """
        if split.type is 'numerical':
            left_indexes = X[:, split.attribute] < split.criteria
            right_indexes = ~left_indexes
        else:
            Z = (
                pd.Index(pd.unique(split.criteria))
                .get_indexer(X[:, split.attribute]))
            left_indexes = np.where(Z >= 0)[0]
            right_indexes = np.where(Z < 0)[0]

        left = X[left_indexes], y[left_indexes]
        right = X[right_indexes], y[right_indexes]

        return left, right

    def _score(
            self, split: Callable[[object], bool], X: np.ndarray,
            y: np.ndarray):
        """
        Calculate the score of split applied to X with respect to y
        """
        def labeling_entropy(
                labeling: np.ndarray, labels: np.ndarray) -> float:
            """
            Calculates the entropy of a labeling given the possible
            labels
            """
            return entropy(
                [
                    sum(pd.Index((label,)).get_indexer(labeling[:, 0]) + 1)
                    / labeling.shape[0]
                    for label in labels
                ],
                base=labels.shape[0])

        (_, yl), (_, yr) = self._split_sample(split, X, y)
        if self.classification:
            # When this tree is being used in a classification context
            # then use classification and split entropies with the
            # mutual information to calculate the score
            classes = np.unique(y)
            l_classes = np.unique(yl)
            r_classes = np.unique(yr)
            l_entropy = labeling_entropy(yl, l_classes)  # Should we use
            r_entropy = labeling_entropy(yr, r_classes)  # `classes` instead?
            l_probability = yl.shape[0]/y.shape[0]
            r_probability = yr.shape[0]/y.shape[0]

            classification_entropy = labeling_entropy(y, classes)
            split_entropy = entropy((l_probability, r_probability), base=2)
            mean_posterior_entropy = (
                l_probability * l_entropy + r_probability * r_entropy)
            mutual_information = (
                classification_entropy - mean_posterior_entropy)
            score = (
                (2 * mutual_information)
                / (classification_entropy + split_entropy))
        else:
            # When this tree is being used in a regression context then
            # use the variance before and after the split is performed
            # to assess the information gain and calculate the score
            total_var = np.var(y)
            l_var = np.var(yl)
            r_var = np.var(yr)

            l_proportion = yl.shape[0] / y.shape[0]
            r_proportion = yr.shape[0] / y.shape[0]

            score = (
                (total_var - l_proportion * l_var - r_proportion * r_var)
                / total_var)

        return score

    def build(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTree':
        """
        Build an Extremely Randomized Decision Tree trained on X, y
        dataset
        :param X: Array of samples for N attributes
        :param y: Array of target values for each sample in X
        :return: self
        """
        def _build(
                X: np.ndarray, y: np.ndarray,
                decision_tree: ExtraTree.DecisionTree):
            # Mark constant attributes
            constant = [
                all(value == sample[0] for value in sample)
                for sample in np.hsplit(X, X.shape[1])]

            if self._stop(X, y, constant):
                if self.classification:
                    count = np.unique(y, return_counts=True)
                    prediction = max(
                        zip(count[0], count[1]),
                        key=lambda t: t[1])[0]
                else:
                    prediction = y.mean()
                leaf = ExtraTree.Leaf(prediction)
                decision_tree.assign(leaf)
                return

            # Select K attributes to draw splits from
            individual_probability = 1 / (X.shape[1] - sum(constant))
            k_choices = np.random.choice(
                X.shape[1],
                min(self.max_features, len(constant) - sum(constant)),
                replace=False,
                p=[
                    individual_probability if not constant[i] else 0
                    for i in range(X.shape[1])
                ])
            K_attributes = X[:, k_choices]

            # Draw random splits
            K_splits = {
                self._pick_random_split(
                    K_attributes[:, column], k_choices[column])
                for column in range(K_attributes.shape[1])}

            # Pick the best split
            split = max(K_splits, key=lambda s: self._score(s, X, y))

            # Branching
            node = ExtraTree.Node(split)
            (Xl, yl), (Xr, yr) = self._split_sample(split, X, y)
            _build(Xl, yl, node.left)
            _build(Xr, yr, node.right)
            decision_tree.assign(node)

        _build(X, y, self.decision_tree)
        return self

    def predict(self, X: np.ndarray):
        return np.asarray([self.decision_tree.predict(x) for x in X])


class ExtraTreeBase(BaseEstimator):
    def __init__(
            self,
            min_samples_split: int = 2,
            max_features: int = None,
            classification: bool = False):
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.classification = classification
        self.tree_ = None  # type: ExtraTree
        self.n_classes_ = None  # type: int
        self.classes_ = None  # type: list
        self.n_outputs_ = None  # type: int
        self.n_features_ = None  # type: int
        self.random_state = None

    def fit(self, X, y, **kwargs):
        # Determine output settings
        n_samples, self.n_features_ = X.shape
        if self.max_features == "auto" or not self.max_features:
            self.max_features = n_samples

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        self.classes_ = [None] * self.n_outputs_
        self.n_classes_ = [1] * self.n_outputs_
        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples))

        # Build tree
        self.tree_ = ExtraTree(
            self.max_features, self.min_samples_split, self.n_classes_,
            self.n_outputs_, self.classification)
        self.tree_.build(X, y)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _validate_X_predict(
            self, X: np.ndarray, check_input: bool) -> np.ndarray:
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError(
                    "No support for np.int64 index based sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must match the input."
                " Model n_features is %s and input n_features is %s "
                % (self.n_features_, n_features))

        return X

    def predict(self, X: np.ndarray, check_input=True):
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        if self.n_outputs_ == 1:
            return proba
        else:
            return proba[:, :, 0]

    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'tree_')
        return self.tree_.compute_feature_importances()


class ExtraTreeRegressor(ExtraTreeBase, RegressorMixin):
    """
    Extremely Randomized Tree Regressor.
    """
    def __init__(self, min_samples_split: int = 2, max_features: int = None):
        super().__init__(min_samples_split, max_features)


class ExtraTreeClassifier(ExtraTreeBase, ClassifierMixin):
    """
    Extremely Randomized Tree Classifier.
    """
    def __init__(self, min_samples_split: int = 2, max_features: int = None):
        super().__init__(min_samples_split, max_features, classification=True)
