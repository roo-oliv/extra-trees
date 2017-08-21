
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

from extra_trees.ensemble.forest import ExtraTreeRegressor

# Create random regression
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regressions model
regr_1 = ExtraTreeRegressor()
regr_2 = DecisionTreeRegressor()
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Print some statitics
print("y_1={}".format(y_1))
print("y_2={}".format(y_2))
"""
print("explained_variance={}".format(explained_variance_score(y_true, y_pred)))
print("mean_absolute_error={}".format(mean_absolute_error(y_true, y_pred)))
print("mean_squared_error={}".format(mean_squared_error(y_true, y_pred)))
print("mean_squared_log_error={}".format(mean_squared_log_error(y_true, y_pred)))
print("median_absolute_error={}".format(median_absolute_error(y_true, y_pred)))
print("r2_score={}".format(r2_score(y_true, y_pred)))
"""

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="extra_tree", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="decision", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Extremely Randomized Tree Regression")
plt.legend()
plt.show()
