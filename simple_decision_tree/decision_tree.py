print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
iris = load_iris()
wine = load_wine()
breast_cancer = load_breast_cancer()
datasets_loaded = [iris, wine, breast_cancer]
datasets_loaded_name = ["iris", "wine", "breast_cancer"]

plot_idx = 1
dataset_count = 0
models = [DecisionTreeClassifier(max_depth=None)]
for dataset_loaded in datasets_loaded:
    dataset_count += 1
    for pair in ([0, 1], [0, 2], [2, 3], [1, 2], [1, 3]):
        for model in models:
            # We only take the two corresponding features
            X = dataset_loaded.data[:, pair]
            y = dataset_loaded.target

            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std

            # Train
            clf = clone(model)
            clf = model.fit(X, y)

            scores = clf.score(X, y)
            # Create a title for each column and the console by using str() and
            # slicing away useless parts of the string
            model_title = str(type(model)).split(
                ".")[-1][:-2][:-len("Classifier")]

            model_details = model_title
            if hasattr(model, "estimators_"):
                model_details += " with {} estimators".format(
                    len(model.estimators_))
            print(model_details + " for " + datasets_loaded_name[dataset_count-1] + " with features", pair,
                  "has a score of", scores)

            plt.subplot(5, 3, plot_idx)
            if plot_idx <= 5*dataset_count:
                # Add a title at the top of each column
                plt.title(datasets_loaded_name[dataset_count-1])

            # Now plot the decision boundary using a fine mesh as input to a
            # filled contour plot
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

            # Plot either a single DecisionTreeClassifier or alpha blend the
            # decision surfaces of the ensemble of classifiers
            if isinstance(model, DecisionTreeClassifier):
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=cmap)
            else:
                # Choose alpha blend level with respect to the number
                # of estimators
                # that are in use (noting that AdaBoost can use fewer estimators
                # than its maximum if it achieves a good enough fit early on)
                estimator_alpha = 1.0 / len(model.estimators_)
                for tree in model.estimators_:
                    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

            # Build a coarser grid to plot a set of ensemble classifications
            # to show how these are different to what we see in the decision
            # surfaces. These points are regularly space and do not have a
            # black outline
            xx_coarser, yy_coarser = np.meshgrid(
                np.arange(x_min, x_max, plot_step_coarser),
                np.arange(y_min, y_max, plot_step_coarser))
            Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                             yy_coarser.ravel()]
                                             ).reshape(xx_coarser.shape)
            cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                    c=Z_points_coarser, cmap=cmap,
                                    edgecolors="none")

            # Plot the training points, these are clustered together and have a
            # black outline
            plt.scatter(X[:, 0], X[:, 1], c=y,
                        cmap=ListedColormap(['r', 'y', 'b']),
                        edgecolor='k', s=20)
            plot_idx += 1  # move on to the next plot in sequence
plt.suptitle("Classifiers on feature subsets")
plt.tight_layout()

plt.show()