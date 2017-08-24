from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

from extra_trees.ensemble.forest import ExtraTreesRegressor

# Create random regression
from extra_trees.tree.tree import ExtraTreeRegressor

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# Models
models = [
    ('min_samples_split=1', ExtraTreeRegressor(min_samples_split=1)),
    ('min_samples_split=2', ExtraTreeRegressor(min_samples_split=2)),
    ('min_samples_split=5', ExtraTreeRegressor(min_samples_split=5)),
    ('min_samples_split=10', ExtraTreeRegressor(min_samples_split=10)),
]

for (name, model), i in zip(models, range(len(models))):
    prediction = model.fit(X, y).predict(X_test)

    plt.figure(i)
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, prediction, color="cornflowerblue", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("ExtraTreeRegressor ({})".format(name))

plt.show()
