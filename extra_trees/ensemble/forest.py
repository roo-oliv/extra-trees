from sklearn.ensemble.forest import ForestRegressor, ForestClassifier

from extra_trees.tree.tree import ExtraTreeRegressor, ExtraTreeClassifier


class ExtraTreesRegressor(ForestRegressor):
    def __init__(
            self,
            n_estimators=10,
            criterion="default",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("max_features", "min_samples_split"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesClassifier(ForestClassifier):
    def __init__(
            self,
            n_estimators=10,
            criterion="default",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("max_features", "min_samples_split"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
