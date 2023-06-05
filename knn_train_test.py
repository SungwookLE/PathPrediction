"""
This module is used for Nearest Neighbor based baselines.
"""

import argparse
import numpy as np
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import time

import utils.baseline_config as baseline_utils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features",
                        default="",
                        type=str,
                        help="path to the file which has train features.")
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
        help="Load the saved model and test")
    parser.add_argument(
        "--use_map",
        action="store_true",
        help="Use the map based features")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used.")
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
    parser.add_argument("--obs_len",
                        default = 20,
                        type = int,
                        help = "Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
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





if __name__ =="__main__":
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
    

    print(data_dict.head()) ########## here (6/5)

    # Perform experiments
    start = time.time()