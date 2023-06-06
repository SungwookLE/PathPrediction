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
    
    def train_and_infer_map(
            self,
            train_input: np.ndarray,
            train_output: np.ndarray,
            test_helpers: pd.DataFrame,
            num_features: int,
            args: Any,
    ) -> None:
        """
        Train and test the model on different prediction horizons for map based Nearest Neighbors.

        Args:
            train_input (numpy array): Train input data
            train_output (numpy array): Train ground truth data
            test_helpers (pandas Dataframe): Test map helpers
            num_features: Number of input features
            args: Arguments passed to runnNNBaselines.py
        
        """
        # Create a temporary directory where forecasted trajectories for all the batches will be saved
        temp_save_dir = tempfile.mkdtemp()

        print(f"Forecasted trajectories will be saved in {args.traj_save_path} ...")

        # Train and Test inputs for kNN
        train_num_tracks = train_input.shape[0]
        train_input = train_input.reshape(
            (train_num_tracks, args.obs_len * num_features), order="F"
        )

        # Train and Test inputs for kNN
        test_centerlines = test_helpers["CANDIDATE_CENTERLINES"].values
        test_nt = test_helpers["CANDIDATE_NT_DISTANCES"].values
        test_references = test_helpers["CANDIDATE_DELTA_REFERENCES"].values
        test_seq_ids = test_helpers["SEQUENCE"].values

        test_num_tracks = test_nt.shape[0]

        for curr_pred_horizon in PREDICTION_HORIZONS:
            grid_search = baseline_utils.get_model(self, train_input, train_output, args, curr_pred_horizon)

            print("Model obtained, now starting inference ...")

            Parallel(
                n_jobs=-2, verbose=5
            )(delayed(self.infer_and_save_traj_map)) #####