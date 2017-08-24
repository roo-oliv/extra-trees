# From MCZA015-13 class notes, modified by Felipe Anchieta Santos Costa
# License: BSD Style.

import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from extra_trees.ensemble.forest import ExtraTreesClassifier


classification_models = [
    ('SVC', SVC()),
    ('RandomForest', RandomForestClassifier()),
    ('ExtraTrees (SciKit)', SKExtraTreesClassifier()),
    ('ExtraTrees', ExtraTreesClassifier()),
]


classification_data_sets = [
    ('breast_cancer', load_breast_cancer(return_X_y=True)),
    ('iris', load_iris(return_X_y=True)),
    ('wine', load_wine(return_X_y=True)),
]


for data_name, data_set in classification_data_sets:

    print("{}\n".format(data_name) + '*' * len(data_name))
    X, y = data_set

    train_size = (len(X) // 4) * 3  # ~75% for training
    test_size = len(X) - train_size # ~25% for testing

    # do some random magic stuff
    fx = np.arange(len(X))
    np.random.shuffle(fx)

    for name, model in classification_models:
        times = []

        print("model: {}\n=======".format(name) + "=" * len(name))
        train = X[fx[0:train_size],:]
        test = X[fx[train_size:],:]

        for _ in range(10):
            start = time.time()
            model.fit(train, y[fx[0:train_size]])
            model.predict(test)
            model.score(test, y[fx[train_size:]])
            end = time.time()
            times.append(end - start)

        print("time mean={}".format(np.mean(times)))
        print("time stdev={}\n".format(np.std(times)))
