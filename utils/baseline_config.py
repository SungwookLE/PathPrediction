"""This module defines all the config parameters."""

FEATURE_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
    "MIN_DISTANCE_FRONT": 6,
    "MIN_DISTANCE_BACK": 7,
    "NUM_NEIGHBORS": 8,
    "OFFSET_FROM_CENTERLINE": 9,
    "DISTANCE_ALONG_CENTERLINE": 10,
}


RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}


# Feature computation
_FEATURES_SMALL_SIZE = 100

# Map Feature computations
_MANHATTAN_THRESHOLD = 5.0  # meters
_DFS_THRESHOLD_FRONT_SCALE = 45.0  # meters
_DFS_THRESHOLD_BACK_SCALE = 40.0  # meters
_MAX_SEARCH_RADIUS_CENTERLINES = 50.0  # meters
_MAX_CENTERLINE_CANDIDATES_TEST = 10


BASELINE_INPUT_FEATURES = {
    "map": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "none": ["X", "Y"],
}

BASELINE_OUTPUT_FEATURES = {
    "map": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "none": ["X", "Y"],
}