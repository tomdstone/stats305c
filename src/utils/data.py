import os
import pickle
from pathlib import Path
import numpy as np


def load_mc_pacman_data():
    """Load the MC Pac-Man pickle.

    Returns:
        Object stored at $DATA/stats-305c-data/mc_pacman.pkl, usually a dict.
    """
    data_path = Path(os.environ["DATA"]) / "stats-305c-data/mc_pacman.pkl"
    with data_path.open("rb") as f:
        return pickle.load(f)


def print_data_structure(data):
    """Print a compact summary of a loaded data object.

    Args:
        data: Expected to be a dict mapping names to arrays, lists, or scalars.
    """
    if not isinstance(data, dict):
        print(f"Data is not a dictionary. Type: {type(data)}")
        return

    print(f"Data structure summary (keys: {list(data.keys())}):")
    for key, value in data.items():
        print(f"\nKey: '{key}'")
        print(f"  Type: {type(value)}")

        if isinstance(value, (list, np.ndarray)):
            length = len(value)
            print(f"  Size/Length: {length}")

            if length > 0:
                # If it's a list of arrays or lists, show their shapes
                if isinstance(value[0], (list, np.ndarray)):
                    sub_type = type(value[0])
                    sub_shapes = [np.shape(x) for x in value[:3]]
                    print(f"  Contains {sub_type} elements.")
                    print(f"  Example shapes: {sub_shapes} ...")
                else:
                    print(f"  Example contents: {value[:5]} ...")
        elif hasattr(value, "shape"):
            print(f"  Shape: {value.shape}")
            print(f"  Example contents: {value[:5]} ...")
        else:
            print(f"  Value: {str(value)[:100]}...")
