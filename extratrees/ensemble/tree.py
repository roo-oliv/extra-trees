import collections
import random
from numbers import Number
from typing import Any, Tuple
from typing import Callable
from typing import List

import numpy
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.tree._tree import issparse
from sklearn.tree.tree import BaseDecisionTree, DTYPE
from sklearn.utils import Bunch, check_array

from extratrees.utils.filter_out_constants import filter_out_constants


def pick_random_split(attribute: list) -> Callable[[Any], bool]:
    if isinstance(attribute[0], Number):
        min_value = min(attribute)
        max_value = max(attribute)
        pivot = random.uniform(min_value+0.0000001, max_value-0.0000001)

        def test(x: Number) -> bool:
            return x < pivot

        test.decision = pivot
        test._type = Number
    else:
        values = set(attribute)
        sample = random.sample(values, random.randint(1, len(values)))

        def test(x: Any) -> bool:
            return x in sample

        test.decision = sample
        test._type = Any

    return test


def split_groups(
        split: Callable[[Any], bool], attributes: numpy.ndarray, feature: int,
        target: numpy.ndarray) -> Tuple[Bunch, Bunch]:
    left_indexes = []
    right_indexes = []
    for row, i, target_value in zip(
            attributes, range(attributes.shape[0]), target):
        value = row[feature]
        if split(value):
            left_indexes.append(i)
        else:
            right_indexes.append(i)

    left = Bunch(
        data=numpy.take(attributes, left_indexes, 0),
        target=numpy.take(target, left_indexes))
    right = Bunch(
        data=numpy.take(attributes, right_indexes, 0),
        target=numpy.take(target, right_indexes))

    return left, right


def score_r(
        split: Callable[[Any], bool], attributes: numpy.ndarray, feature: int,
        target: numpy.ndarray) -> float:
    left, right = split_groups(split, attributes, feature, target)

    total_var = numpy.var(target)  # type: float
    l_var = numpy.var(left.target)  # type: float
    r_var = numpy.var(right.target)  # type: float

    l_proportion = len(left.target) / len(target)
    r_proportion = len(right.target) / len(target)

    return (
        (total_var - l_proportion * l_var - r_proportion * r_var) / total_var)


def score_c(
        split: Callable[[Any], bool], attributes: numpy.ndarray, feature: int,
        target: numpy.ndarray) -> float:
    return score_r(split, attributes, feature, target)
    left, right = split_groups(split, attributes, feature, target)
    split_entropy = (entropy(left.target) + entropy(right.target)) / 2
    mutual_info = mutual_info_score(
        target, attributes[:, feature])  # type: float
    classification_entropy = entropy(attributes[:, feature])  # type: float
    split_entropy = entropy(target)  # type: float

    return (2 * mutual_info) / (classification_entropy + split_entropy)


def stop(
        attributes: numpy.ndarray, features: numpy.ndarray,
        target: numpy.ndarray, min_size: int) -> bool:
    if attributes.shape[0] < min_size:
        return True
    if len(features) == 0:
        return True
    if numpy.unique(target).size == 1:
        return True
    return False


def build_extra_tree(
        attributes: numpy.ndarray, target: numpy.ndarray, min_size: int,
        n_features: int = None, excluded: List[bool] = None,
        _type: type = None):
    dynamic_n_features = False
    if n_features is None:
        dynamic_n_features = True
        n_features = int(numpy.sqrt(attributes.size))
    if excluded is None:
        excluded = [False] * attributes.shape[1]
    if _type is None:
        _type = Number if isinstance(target[0], Number) else Any

    def validate_X_predict(X, check_input):
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if (issparse(X)
                and (X.indices.dtype != numpy.intc
                     or X.indptr.dtype != numpy.intc)):
                raise ValueError(
                    "No support for np.int64 index based sparse matrices")

        X_n_features = X.shape[1]
        if attributes.shape[1] != X_n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (attributes.shape[1], X_n_features))

        return X

    candidates = []
    for i in range(len(excluded)):
        if not excluded[i]:
            candidates.append(i)

    if len(candidates) >= n_features:
        sample = random.sample(candidates, n_features)
        features = numpy.array([row[sample] for row in attributes])
        for feature in sample:
            excluded[feature] = True

        features = filter_out_constants(features, sample)
    else:
        features = numpy.asarray([])

    if stop(attributes, features, target, min_size):
        def predict(data: List[list], show_decisions: bool = False):
            if show_decisions:
                print("LEAF REACHED")
            if _type is Number:
                return sum(target) / float(len(target))
            else:
                return collections.Counter(target)

        predict.decision = None
        predict._validate_X_predict = validate_X_predict

        return predict

    splits = [
        pick_random_split(attribute)
        for attribute in numpy.transpose(features, (1, 0))
    ]
    score = score_r if _type is Number else score_c
    scores = [
        score(split, attributes, feature, target)
        for split, feature in zip(splits, sample)
    ]
    best = numpy.argmax(scores)  # type: int
    split = splits[best]
    attribute_index = sample[best]

    left, right = split_groups(split, attributes, attribute_index, target)
    left_branch = build_extra_tree(
        left.data, left.target, min_size,
        n_features if not dynamic_n_features else None, excluded, _type)
    right_branch = build_extra_tree(
        right.data, right.target, min_size,
        n_features if not dynamic_n_features else None, excluded, _type)

    def predict(data: List[list], show_decisions: bool = False):
        if show_decisions:
            print(split.decision)
        if split(data[attribute_index]):
            return left_branch(data, show_decisions)
        else:
            return right_branch(data, show_decisions)

    predict.feature = attribute_index
    predict.decision = split.decision
    predict._validate_X_predict = validate_X_predict

    return predict
