from extra_trees.ensemble.tree import build_extra_tree

train_data = [
    [1.0, 'blue', 0.8, 30],
    [0.2, 'blue', 0.9, 17],
    [0.3, 'yellow', 0.1, 2],
    [0.7, 'yellow', 0.6, 9],
    [0.9, 'blue', 0.3, 10],
    [1.0, 'yellow', 0.1, 14],
    [0.5, 'yellow', 0.2, 7],
    [0.5, 'blue', 0.5, 19],
    [0.4, 'blue', 0.0, 4],
    [0.1, 'yellow', 0.0, 0],
    [0.3, 'yellow', 0.0, 1],
    [0.3, 'yellow', 0.3, 4],
]

test_data = [
    [1.0, 'blue', 1.0],
    [0.6, 'blue', 0.7],
    [1.0, 'yellow', 1.0],
    [0.0, 'yellow', 1.0],
]


def test_build_extra_tree():
    extra_tree = build_extra_tree(
        [row[:-1] for row in train_data], [row[-1] for row in train_data],
        2, 3)

    predictions = [extra_tree(entry) for entry in test_data]

    assert len(predictions) == len(test_data)  # TODO: real test
