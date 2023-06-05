"""
This module is used for Nearest Neighbor based baselines.
"""

import argparse
import numpy as np
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import time

import utils.baseline_utils as baseline_utils
from utils.knn_utils import Regressor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features."
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features."
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load the saved model and test"
    )
    parser.add_argument(
        "--use_map",
        action="store_true",
        help="Use the map based features"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used."
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position"
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation"
    )
    parser.add_argument(
        "--obs_len",
        default = 20,
        type = int,
        help = "Observed length of the trajectory"
    )
    parser.add_argument(
        "--pred_len",
        default=30,
        type=int,
        help="Prediction Horizon"
    )
    parser.add_argument(
        "--n_neigh",
        default = 1,
        type = int,
        help = "Number of Nearest Neighbors to take, For map-based baselines, it is number of nei"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="path to the pickle file where the model will be / has been saved."
    )
    parser.add_argument(
        "--traj_save_path",
        required=True,
        type=str,
        help="path to the pickle file where forecasted trajectories will be saved."
    )

    return parser.parse_args()


def perform_k_nn_experiments(
        data_dict: Dict[str, Union[np.ndarray, pd.DataFrame, None]],
        baseline_key: str) -> None:
    """
    Perform various experiments using K Nearest Neighbor Regressor

    Args:
        data_dict (dict): Dictionary of train/val/test data
        baseline_key: Key for obtaining features for the baseline
    """

    args = parse_arguments()

    # Get model object for the baseline
    model = Regressor()

if __name__ =="__main__":
    """
    python knn_train_test.py --train_features ./data/train/features/forecasting_features_train.pkl \
                             --val_features ./data/val/features/forecasting_features_val.pkl \
                             --test_features ./data/test_obs/features/forecasting_features_test.pkl \
                             --use_map --use_delta --model_path ./model --traj_save_path ./forecasted_trajectories
    
    """
    args = parse_arguments()

    if not baseline_utils.validate_args(args):
        exit()

    np.random.seed(100)

    # Get features
    if args.use_map:
        baseline_key = "map"
    else:
        baseline_key = "none"
    

    # Get data
    data_dict = baseline_utils.get_data(args, baseline_key)
    
    #print(data_dict) ########## here (6/5)

    # Perform experiments
    start = time.time()