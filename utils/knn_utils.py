"""This module contains utility functions for training and testing Nearest Neighbor based baselines."""

import shutil
import tempfile
from typing import Any, Dict, List, Union

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

import utils.baseline_utils as baseline_utils

PREDICTION_HORIZONS = [30]

class Regressor:
    """
    K-NN Regressor class for training and inference.
    """

    def run_grid_search(self, x: np.ndarray, y: np.ndarray) -> GridSearchCV:
        """
        Run grid search to find best hyperparameters.
        
        Args:
            x: Train input
            y: Train output
        Returns:
            GridSearchCV object
        """

        estimators = [
            ("preprocessing", preprocessing.StandardScaler()),
            ("reduce_dim", PCA())
            ("regressor", KNeighborsRegressor()),
        ]

        pipe = Pipeline(estimators)

        param_grid = dict(
            preprocessing =[None, preprocessing.StandardScaler()],
            reduce_dim = [None, PCA(32)],
            regressor__n_neighbors=[1,8,16],
            regressor__weights=["distance"],
            regressor__n_jobs=[-2],
        )

        custom_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=3
            return_train_score=False,
            verbose=1
        )

        grid_search.fit(x,y)

        return grid_search
    
    