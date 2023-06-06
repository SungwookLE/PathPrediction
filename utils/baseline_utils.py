from collections import OrderedDict
import copy
import math
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union

from argoverse.map_representation.map_api import ArgoverseMap
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate

from utils.baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
)



def viz_predictions(
        input_: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        centerlines: np.ndarray,
        city_names: np.ndarray,
        idx = None,
        show: bool = True,
) -> None:
    """
    Visualize predicted trajectories

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array of list): Top-k predicted trajectory, each with shape (num_tracks x pred_len x 2)
        target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
        ceterlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
        city_names (numpy array): city names for each trajectory
        show (bool): if True, show
    """

    num_tracks = input_.shape[0]
    obs_len = input_.shape[1]
    pred_len = target.shape[1]

    plt.figure(0, figsize = (8,7))
    avm = ArgoverseMap()
    for i in range(num_tracks):
        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
        )
        plt.plot(
            input_[i, -1, 0],
            input_[i, -1, 1],
            "o",
            color="#ECA154",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        plt.plot(
            target[i, :, 0],
            target[i, :, 1],
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
        )
        plt.plot(
            target[i, -1, 0],
            target[i, -1, 1],
            "o",
            color="#d33e4c",
            alpha=1,
            linewidth=3,
            zorder=20,
            markersize=9,
        )

        for j in range(len(centerlines[i])):
            plt.plot(
                centerlines[i][j][:, 0],
                centerlines[i][j][:, 1],
                "--",
                color="grey",
                alpha=1,
                linewidth=1,
                label = f"Ceterline{j}",
                zorder=0,
            )

        for j in range(len(output[i])):
            plt.plot(
                output[i][j][:, 0],
                output[i][j][:, 1],
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                output[i][j][-1, 0],
                output[i][j][-1, 1],
                "o",
                color="#007672",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )

            for k in range(pred_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    output[i][j][k, 0],
                    output[i][j][k, 1],
                    city_names[i],
                    query_search_range_manhattan=2.5,
                )
                #[avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        
        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                input_[i, j, 0],
                input_[i, j, 1],
                city_names[i],
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                target[i, j, 0],
                target[i, j, 1],
                city_names[i],
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        plt.xlim([input_[i, 0, 0]-50, target[i, -1, 0]+50])
        plt.ylim([input_[i, 0, 1]-50, target[i, -1, 1]+50])
        #plt.axis("scaled")

        plt.xticks([])
        plt.yticks([])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend()

        if show:
            plt.show(block=False)
            plt.pause(5)
            plt.close()

def validate_args(args: Any) -> bool:
    """Validate the arguments passed to the baseline.

    Args:
        args: Arguments to the baselines.

    Returns:
        success: True if args valid.

    """
    success = True
    if args.normalize and args.use_map:
        print(
            "[ARGS ERROR]: normalize and use_map cannot be used simultaneously."
        )
        success = False
   
    if args.obs_len > 20:
        print("[ARGS ERROR]: obs_len cannot be more than 20.")
        success = False
    if args.pred_len > 30:
        print("[ARGS ERROR]: pred_len cannot be more than 30.")
        success = False
    return success

def get_normalized_traj(df: pd.DataFrame, args: Any) -> np.ndarray:
    """
    Normalize trajectory such that it starts at (0,) and observed part ends on x-axis
    
    Args:
        df (pandas DataFrame): Data for all the tracks
        args: Arguments passed to the baseline code
    Returns:
        normalize_traj_arr (numpy array): Array of shape (num_tracks x seq_len x 2)
                                          containing normalized trajectory
    Note:
        This also updates the dataframe in-place
    """
    # Transformation values will be saved in df
    translation = []
    rotation = []

    normalized_traj = []
    x_coord_seq = np.stack(df["FEATURES"].values)[:, :, FEATURE_FORMAT["X"]]
    y_coord_seq = np.stack(df["FEATURES"].values)[:, :, FEATURE_FORMAT["Y"]]

    # Normalize each trajectory
    for i in range(x_coord_seq.shape[0]):
        xy_seq = np.stack((x_coord_seq[i], y_coord_seq[i]), axis=1)

        start = xy_seq[0]

        # First apply translation
        m = [1, 0, 0, 1, -start[0], -start[1]]
        ls = LineString(xy_seq)

        # Now apply rotation, takng care of edge cases
        ls_offset = affine_transform(ls, m)
        end = ls_offset.coords[args.obs_len -1]
        if end[0] == 0 and end[1] == 0:
            angle = 0.0
        elif end[0] == 0:
            angle = -90.0 if end[1] > 0 else 90.0 # (6/5, 부호가 이게 맞나?)
        elif end[1] == 0:
            angle = 0.0 if end[0] > 0 else 180.0
        else:
            angle = math.degrees(math.atan(end[1] / end[0]))
            if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
                angle = -angle
            else:
                angle = 180 - angle
        
        # Rotate the trajectory
        ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]

        # Normalized trajectory
        norm_xy = np.array(ls_rotate)

        # Update the containers
        normalized_traj.append(norm_xy)
        translation.append(m)
        rotation.append(angle)

    # Update the dataframe and return the normalized trajectory
    normalized_traj_arr = np.stack(normalized_traj)
    df["TRANSLATION"] = translation
    df["ROTATION"] = rotation
    return normalized_traj_arr

def get_relative_distance(data: np.ndarray, mode: str, args: Any) -> np.ndarray:
    """
    Convert absolute distance to relative distance in place and return the reference (first value).

    Args:
        data (numpy array): Data array of shape (num_tracks x seq_len x num_features). Distances arw always the first 2 features
        mode: train/val/test
        args: Arguments pass to the baseline code
    Returns:
        reference (numpy array): First value of the sequence of data with shape (num_tracks x 2). For map based baselines, it will be first n-t distance of the trajectory.
    """

    reference = copy.deepcopy(data[:, 0, :2])

    if mode == "test":
        traj_len = args.obs_len
    else:
        traj_len = args.obs_len + args.pred_len
    
    for i in range(traj_len -1, 0, -1):
        data[:, i, :2] = data[:, i, :2] - data[:, i-1, :2]
    data[:, 0, :] = 0

    return reference

def load_and_preprocess_data(
        input_features: List[str],
        output_features: List[str],
        args: Any,
        feature_file: str,
        mode: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the data and preprocess based on given arguments.

    Args:
        input_features (list of str): Input features for the baseline
        output features (list of str): Output features for the baseline
        args (argparse): Arguments to runBaselines.py
        feature_file: path to the file containing features
        mode (str): train/val/test
    
    Returns:
        _input: Input to the baseline
        _output: Ground truth
        df: Helper values useful in visualization and evaluation
    """
    df = pd.read_pickle(feature_file)

    # Normalize if its a non-map baseline
    if not args.use_map and args.normalize:
        print("Normalizing ...")

        # Don't use X,Y as features
        input_feature_idx = [
            FEATURE_FORMAT[feature] for feature in input_features
            if feature != "X" and feature != "Y"
        ]
        output_feature_idx = [
            FEATURE_FORMAT[feature] for feature in output_features
            if feature != "X" and feature != "Y"
        ]

        # Normalize the trajectory
        normalized_traj_arr = get_normalized_traj(df, args)

        # Get other features
        input_features_data = np.stack(
            df["FEATURES"].values)[:, :, input_feature_idx].astype("float")
        output_features_data = np.stack(
            df["FEATURES"].values)[:, :, output_feature_idx].astype("float")

        # Merge normalized trajectory and other features
        input_features_data = np.concatenate(
            (normalized_traj_arr, input_features_data), axis =2
        )        
        output_features_data = np.concatenate(
            (normalized_traj_arr, output_features_data), axis=2
        )
    else:
        input_feature_idx = [
            FEATURE_FORMAT[feature] for feature in input_features
        ]
        output_feature_idx = [
            FEATURE_FORMAT[feature] for feature in output_features
        ]

        input_features_data = np.stack(
            df["FEATURES"].values)[:, :, input_feature_idx].astype("float")
        output_features_data = np.stack(
            df["FEATURES"].values)[:, :, output_feature_idx].astype("float")

    # If using relative distance instead of absolute
    # Store the first coordinate (reference) of the trajectory to map it bat to absolute values later
    if args.use_delta:

        # Get relative distances for all topk centerline candidates
        if args.use_map and mode == "test":
            print("Creating relative distances for candidate centerlines ...")

            # Relative candidate distances nt
            candidate_nt_distances = df["CANDIDATE_NT_DISTANCES"].values
            candidate_references = []
            for candidate_nt_dist_i in candidate_nt_distances:
                curr_reference = []
                for curr_candidate_nt in candidate_nt_dist_i:
                    curr_candidate_reference = get_relative_distance(
                        np.expand_dims(curr_candidate_nt, 0), mode, args)

                    curr_candidate_nt = curr_candidate_nt.squeeze()
                    curr_reference.append(curr_candidate_reference.squeeze())
                
                candidate_references.append(curr_reference)

            df["CANDIDATE_DELTA_REFERENCES"] = candidate_references

        else:
            print("Creating relative distances...")

            # Relative features
            reference = get_relative_distance(input_features_data, mode, args)
            _ = get_relative_distance(output_features_data, mode, args)
            df["DELTA_REFERENCE"] = reference.tolist()
        
    # Set train and test input/output data
    _input = input_features_data[:, :args.obs_len]

    if mode == "test":
        _output = None
    else:
        _output = output_features_data[:, args.obs_len:]
    
    return _input, _output, df



def get_data(args: Any
             , baseline_key: str) -> Dict[str,Union[np.ndarray, pd.DataFrame, None]]:
    """
    Load data from local data_dir.

    Args:
        args (argparse): Arguments to baseline
        baseline_key: Key for obtaining features for the baseline
    Returns:
        data_dict (dict): Dictionary of input/output data and helpers for train/val/test splits
    
    """
    input_features = BASELINE_INPUT_FEATURES[baseline_key]
    output_features = BASELINE_OUTPUT_FEATURES[baseline_key]
    if args.test_features:
        print("Loading Test data ...")
        test_input, test_output, test_df = load_and_preprocess_data(
            input_features,
            output_features,
            args,
            args.test_features,
            mode = "test"
        )
        print(f"Test size: {test_input.shape[0]}")
    else:
        test_input, test_output, test_df = [None] *3
    
    if args.train_features:
        print("Loading Train data ...")
        train_input, train_output, train_df = load_and_preprocess_data(
            input_features,
            output_features,
            args,
            args.train_features,
            mode="train")
        print("Train Size: {}".format(train_input.shape[0]))
    else:
        train_input, train_output, train_df = [None] * 3

    if args.val_features:
        print("Loading Val data ...")
        val_input, val_output, val_df = load_and_preprocess_data(
            input_features,
            output_features,
            args,
            args.val_features,
            mode="val")
        print("Val Size: {}".format(val_input.shape[0]))
    else:
        val_input, val_output, val_df = [None] * 3
    
    data_dict = {
        "train_input": train_input,
        "val_input": val_input,
        "test_input": test_input,
        "train_output": train_output,
        "val_output": val_output,
        "test_output": test_output,
        "train_helpers": train_df,
        "val_helpers": val_df,
        "test_helpers": test_df,
    }

    return data_dict

def get_model(
        regressor: Any,
        train_input: np.ndarray,
        train_output: np.ndarray,
        args: Any,
        pred_horizon: int,
)-> Any:
    """
    Get the trained model after running grid search or load a saved one.

    Args:
        regressor: Nearest Neighbor regressor class instance
        train_input: Input to the model
        train_output: Ground truth for the model
        args: Arguments passed to the baseline
        pred_horizon: Prediction Horizon

    Returns:
        grid_search: sklearn GridSearchCV object
    
    """

    # Load model
    if args.test:

        # Load a trained model
        with open(args.model_path, "rb") as f:
            grid_search = pkl.load(f)
        print(f"## Loaded {args.model_path} ...")
    
    else:
        train_num_tracks = train_input.shape[0]

        # Flatten to (num_tracks x feature_size)
        train_output_curr = train_output[:, :pred_horizon, :].reshape(
            (train_num_tracks, pred_horizon*2), ordef="F"
        )

        # Run grid search for hyper parameter tuning
        grid_search = regressor.run_grid_search(train_input, train_output_curr)
        os.makedirs(os.path.dirname(args.model_path), exist_ok = True)
        with open(args.model_path, "wb") as f:
            pkl.dump(grid_search, f)
        print(f"Trained model saved at... {args.model_path}")
    
    return grid_search