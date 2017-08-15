import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from extratrees.ensemble.forest import ExtraTreesClassifier

n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # Superficie de Contorno
plot_step_coarser = 0.5  # Largura do Passo
RANDOM_SEED = 13  # Seed em cada iteracao

iris_dataset = load_iris()
wine_dataset = load_wine()
breast_cancer_dataset = load_breast_cancer()
datasets = {
    "iris": iris_dataset,
    "wine": wine_dataset,
    "breast_cancer": breast_cancer_dataset
}

plot_idx = 1
dataset_count = 0
models = [
    DecisionTreeClassifier(max_depth=None),
    ExtraTreesClassifier(n_estimators, 3, 4)
]
for dataset_name, dataset in datasets.items():
    dataset_count += 1
    for model in models:
        X = dataset.data
        y = dataset.target

        # randomize
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # normalize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # train
        clf = model.fit(X, y)

        scores = clf.score(X, y)

        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))

        print(
            model_details + " for " + dataset_name
            + " with all features has a score of", scores)

        plt.subplot(5, 3, plot_idx)
        if plot_idx <= 5*dataset_count:
            plt.title(dataset_name)

        # decision limit
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, plot_step),
            np.arange(y_min, y_max, plot_step))

        # plot contour
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(
            np.c_[xx_coarser.ravel(), yy_coarser.ravel()]
        ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(
            xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap,
            edgecolors="none")

        # plot training points
        plt.scatter(
            X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'y', 'b']),
            edgecolor='k', s=20)
        plot_idx += 1

plt.suptitle("Classifiers on feature subsets")
plt.tight_layout()

plt.show()
