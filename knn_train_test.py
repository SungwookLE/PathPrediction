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

    test_input = data_dict["test_input"]
    test_output = data_dict["test_output"]
    test_helpers = data_dict["test_helpers"]

    train_input = data_dict["train_input"]
    train_output = data_dict["train_output"]
    train_helpers = data_dict["train_helpers"]

    val_input = data_dict["val_input"]
    val_output = data_dict["val_output"]
    val_helpers = data_dict["val_helpers"]

    # Merge train and val splits and use K-fold cross validation instead
    train_val_input = np.concatenate((train_input, val_input))
    train_val_output = np.concatenate((train_output, val_output))
    train_val_helpers = np.concatenate([train_helpers, val_helpers])

    if args.use_map:
        print("####  Training Nearest Neighbor in NT frame  ###")
        model.train_and_infer_map(
            train_val_input,
            train_val_output,
            test_helpers,
            len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
            args,
        )

    else:
        # not yet considered (6/7)
        pass

if __name__ =="__main__":
    """
    python knn_train_test.py --train_features ./data/train/features/forecasting_features_train.pkl \
                             --val_features ./data/val/features/forecasting_features_val.pkl \
                             --test_features ./data/test_obs/features/forecasting_features_test.pkl \
                             --use_map --use_delta --model_path ./model/knn.pth --traj_save_path ./forecasted_trajectories/knn_test.pkl
    
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

    # Perform experiments (Train)
    # start = time.time()
    # perform_k_nn_experiments(data_dict, baseline_key)
    # end = time.time()
    # print(f"Completed experiment in {(end-start)/60.0} mins")


    # for visualization (6/8)
    from utils.baseline_utils import viz_predictions
    from utils.baseline_config import FEATURE_FORMAT
    import re

    feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
    df = data_dict["test_helpers"]

    pred_trajectories = pd.read_pickle(args.traj_save_path)
    seq_id = df["SEQUENCE"].values

    gt_trajectory = np.stack(
        df["FEATURES"].values)[:, -args.pred_len:, feature_idx].astype("float")
    
    gt_trajectories = dict()
    for i in range(gt_trajectory.shape[0]):
        gt_trajectories[seq_id[i]] = gt_trajectory[i]

    seq_ids = pred_trajectories.keys()
    
    for seq_id in seq_ids:
        gt_trajectory = gt_trajectories[seq_id]
        curr_features_df = df[df["SEQUENCE"] == seq_id]
        input_trajectory = (
            curr_features_df["FEATURES"].values[0][:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype("float")
        )
        output_trajectories = pred_trajectories[seq_id]

        train_test_flag = re.split(r"[_.]", args.test_features)[-2]
        if train_test_flag == "test":
            candidate_centerlines = curr_features_df["CANDIDATE_CENTERLINES"].values[0]
        elif train_test_flag == "train":
            candidate_centerlines = [curr_features_df["ORACLE_CENTERLINE"].values[0]]
        
        city_name = curr_features_df["FEATURES"].values[0][0, FEATURE_FORMAT["CITY_NAME"]]

        gt_trajectory = np.expand_dims(gt_trajectory, 0)
        input_trajectory = np.expand_dims(input_trajectory, 0)
        output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
        candidate_centerlines = np.expand_dims(np.array(candidate_centerlines), 0)

        city_name = np.array([city_name])

        viz_predictions(
            input_trajectory,
            output_trajectories,
            gt_trajectory,
            candidate_centerlines,
            city_name,
            show=True
        )

