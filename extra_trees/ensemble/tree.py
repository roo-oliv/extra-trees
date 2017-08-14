import collections
import copy
import random
from numbers import Number
from typing import Any, Tuple
from typing import Callable
from typing import List

import numpy
from sklearn.utils import Bunch

from extra_trees.utils.filter_out_constants import filter_out_constants


def pick_random_split(attribute: list) -> Callable[[Any], bool]:
    if isinstance(attribute[0], Number):
        min_value = min(attribute)
        max_value = max(attribute)
        pivot = random.uniform(min_value, max_value)

        def test(x: Number) -> bool:
            return x < pivot

        test._type = Number
    else:
        values = set(attribute)
        sample = random.sample(values, random.randint(1, len(values)))

        def test(x: Any) -> bool:
            return x in sample

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
    return 1  # TODO


def stop(
        attributes: numpy.ndarray, features: numpy.ndarray,
        target: numpy.ndarray, min_size: int) -> bool:
    if attributes.shape[1] < min_size:
        return True
    if len(features) == 0:
        return True
    if numpy.unique(target).size == 1:
        return True
    return False


def build_extra_tree(
        train: Bunch, n_features: int, min_size: int,
        excluded: List[bool] = None, _type: type = None):
    attributes = train.data
    target = train.target

    if excluded is None:
        excluded = [False] * attributes.shape[1]
    if _type is None:
        _type = Number if isinstance(target[0], Number) else Any

    candidates = []
    for i in range(len(excluded)):
        if not excluded[i]:
            candidates.append(i)

    if len(candidates) >= n_features:
        sample = random.sample(candidates, n_features)
        features = numpy.array([row[sample] for row in attributes])
        for feature in sample:
            excluded[feature] = True

        features = filter_out_constants(features)
    else:
        features = numpy.ndarray([0])

    if stop(attributes, features, target, min_size):
        def predict(data):
            if _type is Number:
                return sum(target) / float(len(target))
            else:
                return collections.Counter(target)

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
    best = numpy.argmax(scores)
    split = splits[best]
    attribute = features[best]
    attribute_index = candidates[best]

    left, right = split_groups(split, attributes, attribute_index, target)
    left_branch = build_extra_tree(
        left, n_features, min_size, excluded, _type)
    right_branch = build_extra_tree(
        right, n_features, min_size, excluded, _type)

    def predict(data: List[list]):
        if split(data[attribute_index]):
            return left_branch(data)
        else:
            return right_branch(data)

    return predict
