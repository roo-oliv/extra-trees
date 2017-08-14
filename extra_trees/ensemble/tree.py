import collections
import copy
import random
from numbers import Number
from typing import Any, Tuple
from typing import Callable
from typing import List

import numpy


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
        split: Callable[[Any], bool], attribute: list, attributes: List[list],
        target: list) \
        -> Tuple[Tuple[List[list], list], Tuple[List[list], list]]:
    left = ([], [])
    right = ([], [])
    for value, row, target_value in attribute, attributes, target:
        if split(value):
            left[0].append(copy.deepcopy(row))
            left[1].append(copy.deepcopy(target_value))
        else:
            right[0].append(copy.deepcopy(row))
            right[1].append(copy.deepcopy(target_value))

    return left, right


def score_r(
        split: Callable[[Any], bool], attribute: list, attributes: List[list],
        target: List[Number]) -> float:
    return 1  # TODO


def score_c(
        split: Callable[[Any], bool], attribute: list, attributes: List[list],
        target: List[Any]) -> float:
    return 1  # TODO


def stop(
        features: List[list], target: list, min_size: int) -> bool:
    if len(features) < min_size:
        return True
    if all(x.count(x[0]) == len(x) for x in features):
        return True
    if target.count(target[0]) == len(target):
        return True
    return False


def build_extra_tree(
        attributes: List[list], target: list, n_features: int, min_size: int,
        excluded: List[bool] = None, _type: type = None):
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

    if stop(features, target, min_size):
        def predict(data):
            if _type is Number:
                return sum(target) / float(len(target))
            else:
                return collections.Counter(target)

        return predict

    splits = [pick_random_split(attribute) for attribute in features]
    score = score_r if _type is Number else score_c
    scores = [score(split, features, target) for split in splits]
    best = numpy.argmax(scores)
    split = splits[best]
    attribute = features[best]
    attribute_index = candidates[best]

    left, right = split_groups(split, attribute, attributes, target)
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
