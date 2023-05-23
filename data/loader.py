import os
from typing import Tuple

import numpy as np

from astrophysics.data import npz_dir, combined_dataset


def load_combined_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = np.load(os.path.join(npz_dir, combined_dataset["matrices"]))["matrices"]
    features = np.load(os.path.join(npz_dir, combined_dataset["features"]))["features"]

    # истинные значения, на основе которых моделировались события
    true_features = np.load(os.path.join(npz_dir, combined_dataset["true_features"]))[
        "true_features"
    ]

    return matrices, features, true_features
