"""
const_vel_train_test.py runs a constant velocity baseline.
"""

import argparse
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import mean_squared_error

from utils.baseline_config import FEATURE_FORMAT
import utils.baseline_config as baseline_utils
import re

def parse_arguments():
    """Parse Arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--traj_save_path",
        required=True,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()

def get_mean_velocity(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get man velocity of the observed trajectory

    Args:
        coords: Coordinates for the trajectory
    Returns:
        Mean velocity along x and y
    """

    #coords.shape = (100,20,2)
    vx, vy = (
        np.zeros((coords.shape[0], coords.shape[1] -1)), # vx.shape = (100, 19)
        np.zeros((coords.shape[0], coords.shape[1] -1)), # vy.shape = (100, 19)
    )

    for i in range(1, coords.shape[1]):
        vx[:, i-1] = (coords[:, i, 0] - coords[:, i-1, 0]) / 0.1 
        vy[:, i-1] = (coords[:, i, 1] - coords[:, i-1, 1]) / 0.1
    
    vx = np.mean(vx, axis=1)
    vy = np.mean(vy, axis=1)

    return vx, vy

def predict(obs_trajectory: np.ndarray, vx: np.ndarray, vy: np.ndarray,
            args: Any) -> np.ndarray:
    """
    Predict future trajectory given mean velocity.

    Args:
        obs_trajectory: Observed Trajectory
        vx: Mean velocity along x
        vy: Mean velocity along y
        args: Arguments to the baseline
    
    Returns:
        pred_trajectory: Future trajectory
         
    """
    pred_trajectory = np.zeros((obs_trajectory.shape[0], args.pred_len, 2)) # pred_trajectory.shape = (100, 30, 2)
    

    prev_coords = obs_trajectory[:, -1, :]
    for i in range(args.pred_len):
        pred_trajectory[:, i, 0] = prev_coords[:, 0] + vx*0.1
        pred_trajectory[:, i, 1] = prev_coords[:, 1] + vy*0.1
        prev_coords = pred_trajectory[:,i]
    
    return pred_trajectory


def forecast_and_save_trajectory(obs_trajectory: np.ndarray,
                                 seq_id: np.ndarray, args: Any) -> None:
    """
    Forecast future trajectory and save it.

    Args:
        obs_trajectory: Observed trajectory
        seq_id: Sequence ids
        args: Arguments to the baseline

    """

    vx, vy = get_mean_velocity(obs_trajectory)
    pred_trajectory = predict(obs_trajectory, vx, vy, args)

    forecasted_trajectories = {}
    for i in range(pred_trajectory.shape[0]): # for i in range(100)
        forecasted_trajectories[seq_id[i]] = [pred_trajectory[i]] 
    
    with open(args.traj_save_path, "wb") as f:
        pkl.dump(forecasted_trajectories, f)

if __name__ == "__main__":
    # python const_vel_train_test.py --test_features ./data/train/features/forecasting_features_train.pkl --obs_len 20 --pred_len 30 --traj_save_path forecasted_trajectories/const_vel_train.pkl
    # python const_vel_train_test.py --test_features ./data/test_obs/features/forecasting_features_test.pkl --obs_len 20 --pred_len 30 --traj_save_path forecasted_trajectories/const_vel_test.pkl

    args = parse_arguments()
    df = pd.read_pickle(args.test_features)

    feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]] 
    seq_id = df["SEQUENCE"].values

    obs_trajectory = np.stack(
        df["FEATURES"].values)[:, :args.obs_len, feature_idx].astype("float") # 총 데이터 seq_id 의 obs_len 만큼 잘른 (X,Y)
    forecast_and_save_trajectory(obs_trajectory, seq_id, args)
    

    ### for visualization (6/4)
    from utils.baseline_utils import viz_predictions

    pred_trajectories = pd.read_pickle(args.traj_save_path)
    gt_trajectory = np.stack(
        df["FEATURES"].values)[:, -args.pred_len:, feature_idx].astype("float")
    gt_trajectories = {}
    for i in range(gt_trajectory.shape[0]): # for i in range(100)
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
            show=True,
        )