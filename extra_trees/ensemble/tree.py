import collections
import random
from numbers import Number
from typing import Any
from typing import Callable
from typing import List

import numpy


def pick_random_split(attribute: list) -> function:
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


def score_r(
        split: Callable[[Any], bool], attributes: List[list],
        target: List[Number]) -> float:
    return 1  # TODO


def score_c(
        split: Callable[[Any], bool], attributes: List[list],
        target: List[Any]) -> float:
    return 1  # TODO


def stop(
        attributes: List[list], target: list, min_size: int) -> bool:
    if len(attributes) < min_size:
        return True
    if all(x.count(x[0]) == len(x) for x in attributes):
        return True
    if target.count(target[0]) == len(target):
        return True
    return False


def build_extra_tree(
        attributes: List[list], target: list, n_features: int, min_size: int,
        _type: type = None):
    if _type is None:
        _type = Number if isinstance(target[0], Number) else Any

    if stop(attributes, target, min_size):
        if _type is Number:
            return sum(target) / float(len(target))
        else:
            return collections.Counter(target)

    features = []
    for _ in range(n_features):
        i = random.randrange(len(attributes))
        features.append(attributes.pop(i))
        del target[i]

    splits = [pick_random_split(attribute) for attribute in features]
    score = score_r if _type is Number else score_c
    scores = [score(split, attributes, target) for split in splits]
    split = splits[numpy.argmax(scores)]

    # TODO

