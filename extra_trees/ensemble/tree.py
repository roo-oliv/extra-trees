import collections
import copy
import random
from numbers import Number
from typing import Any, Tuple
from typing import Callable
from typing import List

import numpy
from sklearn.utils import Bunch


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
        split: Callable[[Any], bool], attributes: List[list], feature: int,
        target: list) -> Tuple[Bunch, Bunch]:
    left = Bunch(data=[], target=[])
    right = Bunch(data=[], target=[])
    for value, row, target_value in attributes[feature], attributes, target:  # FIXME: Broken use of attributes
        if split(value):
            left.data.append(copy.deepcopy(row))
            left.target.append(copy.deepcopy(target_value))
        else:
            right.data.append(copy.deepcopy(row))
            right.target.append(copy.deepcopy(target_value))

    return left, right


def score_r(
        split: Callable[[Any], bool], attributes: List[list], feature: int,
        target: List[Number]) -> float:
    left, right = split_groups(split, attributes, target)

    total_var = numpy.var(target)  # type: float
    l_var = numpy.var(left.target)  # type: float
    r_var = numpy.var(right.target)  # type: float

    l_proportion = len(left.target) / len(attribute)  # FIXME attributes?
    r_proportion = len(right.target) / len(attribute)

    return (
        (total_var - l_proportion * l_var - r_proportion * r_var) / total_var)


def score_c(
        split: Callable[[Any], bool], attributes: List[list], feature: int,
        target: List[Any]) -> float:
    return 1  # TODO


def stop(
        attributes: List[list], features: List[list], target: list,
        min_size: int) -> bool:
    if len(attributes) < min_size:
        return True
    if all(numpy.unique(x).size == 1 for x in features):
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
        excluded = [False] * len(attributes)
    if _type is None:
        _type = Number if isinstance(target[0], Number) else Any

    candidates = []
    for i in range(len(excluded)):
        if not excluded[i]:
            candidates.append(i)

    sample = random.sample(candidates, n_features)
    features = [attributes[feature] for feature in sample]
    for feature in sample:
        excluded[feature] = True

    if stop(attributes, features, target, min_size):
        def predict(data):
            if _type is Number:
                return sum(target) / float(len(target))
            else:
                return collections.Counter(target)

        return predict

    splits = [pick_random_split(attribute) for attribute in features]
    score = score_r if _type is Number else score_c
    scores = [
        score(split, attributes, feature, target)
        for split, feature in zip(splits, features)
    ]
    best = numpy.argmax(scores)
    split = splits[best]
    attribute = features[best]
    attribute_index = candidates[best]

    left, right = split_groups(split, attribute, target, attributes)
    left_branch = build_extra_tree(
        *left, n_features, min_size, excluded, _type)
    right_branch = build_extra_tree(
        *right, n_features, min_size, excluded, _type)

    def predict(data: List[list]):
        if split(data[attribute_index]):
            return left_branch(data)
        else:
            return right_branch(data)

    return predict
