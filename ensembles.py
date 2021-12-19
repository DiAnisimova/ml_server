import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from time import time



class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        self.models = []
        self.ind = []
        self.rmse = []
        self.rmse_test = []
        self.time = []
        start = time()
        predictions = np.zeros((y.shape))
        if y_val is not None:
            predictions_test = np.zeros((y_val.shape))
        for i in range(self.n_estimators):
            ind_features = np.random.choice(X.shape[1], size=self.feature_subsample_size, replace=False)
            ind_obj = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
            model.fit(X[ind_obj,:][:, ind_features], y[ind_obj])
            self.models.append(model)
            self.ind.append(ind_features)
            pred_train = model.predict(X[:, ind_features])
            predictions += pred_train
            self.rmse.append((np.average((y - predictions/ (i + 1)) ** 2, axis=0)) ** 0.5)
            if X_val is not None:
                pred_test = model.predict(X_val[:, ind_features])
                predictions_test += pred_test
                self.rmse_test.append((np.average((y_val - predictions_test/ (i + 1)) ** 2, axis=0)) ** 0.5)               
            self.time.append(time() - start)  
            

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(X[:, self.ind[i]]))
        return np.mean(res, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        
    

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        def loss(alpha, prev, b, y):
            return ((prev + alpha * b - y) ** 2).sum()
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        self.models = []
        self.ind = []
        self.rmse = []
        self.rmse_test = []
        self.time = []
        self.alphas = []
        start = time()
        predictions = np.zeros((y.shape))
        if y_val is not None:
            predictions_test = np.zeros((y_val.shape))
        for i in range(self.n_estimators):
            ind_features = np.random.choice(X.shape[1], size=self.feature_subsample_size, replace=False)
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
            model.fit(X[:, ind_features], y - predictions)
            self.models.append(model)
            self.ind.append(ind_features)
            pred_train = model.predict(X[:, ind_features])
            cur_alpha = minimize_scalar(loss, args=(predictions, pred_train, y)).x
            self.alphas.append(cur_alpha)
            predictions += cur_alpha * self.learning_rate * pred_train
            self.rmse.append(np.average((y - predictions) ** 2, axis=0) ** 0.5)
            if X_val is not None:
                pred_test = model.predict(X_val[:, ind_features])
                predictions_test += cur_alpha * self.learning_rate * pred_test
                self.rmse_test.append(np.average((y_val - predictions_test) ** 2, axis=0) ** 0.5)               
            self.time.append(time() - start)
            

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(X[:, self.ind[i]]))
        return (np.array(res) * np.array(self.alphas)[:, None] ).sum(axis=0) * self.learning_rate