from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from astrophysics.data.utils import to_XY_astropy

plt.style.use("ggplot")


def scale_matrices(
    matrices: np.ndarray, scaler_time=None, scaler_edep=None, scaler_muons=None
):
    scaled = deepcopy(matrices)

    if scaler_time is None:
        scaler_time, scaler_edep, scaler_muons = (
            MinMaxScaler(),
            StandardScaler(),
            StandardScaler(),
        )

        scaled_times = scaler_time.fit_transform(
            scaled[..., 0].flatten().reshape(-1, 1)
        )
        scaled_edep = scaler_edep.fit_transform(
            np.log(scaled[..., 1].flatten().reshape(-1, 1) + 1)
        )
        scaled_muons = scaler_muons.fit_transform(
            scaled[..., 2].flatten().reshape(-1, 1)
        )
    else:
        scaled_times = scaler_time.transform(scaled[..., 0].flatten().reshape(-1, 1))
        scaled_edep = scaler_edep.transform(
            np.log(scaled[..., 1].flatten().reshape(-1, 1) + 1)
        )
        scaled_muons = scaler_muons.transform(scaled[..., 2].flatten().reshape(-1, 1))

    scaled[..., 0] = scaled_times.reshape(matrices.shape[0], 16, 16)
    scaled[..., 1] = scaled_edep.reshape(matrices.shape[0], 16, 16)
    scaled[..., 2] = scaled_muons.reshape(matrices.shape[0], 16, 16)

    return scaled, scaler_time, scaler_edep, scaler_muons


def split(matrices: np.ndarray, features: np.ndarray, target: np.ndarray):
    (
        matrices_train,
        matrices_test,
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        matrices, features, target, test_size=0.2, stratify=target, random_state=21
    )

    (
        matrices_train,
        matrices_valid,
        features_train,
        features_valid,
        target_train,
        target_valid,
    ) = train_test_split(
        matrices_train,
        features_train,
        target_train,
        test_size=0.25,
        stratify=target_train,
        random_state=21,
    )

    return (
        matrices_train,
        matrices_valid,
        matrices_test,
        features_train,
        features_valid,
        features_test,
        target_train,
        target_valid,
        target_test,
    )


def build_augmentations(
    matrices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices_rot_1 = np.rot90(matrices, 1, axes=(1, 2))
    matrices_rot_2 = np.rot90(matrices, 2, axes=(1, 2))
    matrices_rot_3 = np.rot90(matrices, 3, axes=(1, 2))

    return matrices_rot_1, matrices_rot_2, matrices_rot_3


def plot_augmentations(
    matrices: np.ndarray,
    matrices_rot_1: np.ndarray,
    matrices_rot_2: np.ndarray,
    matrices_rot_3: np.ndarray,
):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(21, 7), dpi=256)
    sns.heatmap(matrices[0, ..., 0], square=True, ax=ax0)
    sns.heatmap(matrices_rot_1[0, ..., 0], square=True, ax=ax1)
    sns.heatmap(matrices_rot_2[0, ..., 0], square=True, ax=ax2)
    sns.heatmap(matrices_rot_3[0, ..., 0], square=True, ax=ax3)

    plt.tight_layout()
    plt.savefig("aug.png")

    plt.show()


def rotate_x_y_Az(features_vector, n90):
    features_vector = features_vector.copy()

    if n90 == 1 or n90 == 2:
        features_vector[1] = -features_vector[1]

    if n90 == 2 or n90 == 3:
        features_vector[2] = -features_vector[2]

    features_vector[5] += n90 * 90

    if features_vector[5] < 0:
        features_vector[5] += 360

    if features_vector[5] > 360:
        features_vector[5] -= 360

    return features_vector


def build_rotate_features(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_rot_1 = np.apply_along_axis(rotate_x_y_Az, 1, features, n90=1)
    features_rot_2 = np.apply_along_axis(rotate_x_y_Az, 1, features, n90=2)
    features_rot_3 = np.apply_along_axis(rotate_x_y_Az, 1, features, n90=3)

    return features_rot_1, features_rot_2, features_rot_3


def add_augmentations(
    matrices: np.ndarray,
    matrices_rot_1: np.ndarray,
    matrices_rot_2: np.ndarray,
    matrices_rot_3: np.ndarray,
    features: np.ndarray,
    features_rot_1: np.ndarray,
    features_rot_2: np.ndarray,
    features_rot_3: np.ndarray,
    target: np.ndarray,
    aug_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = features.shape[0]

    choice1 = np.random.randint(size, size=int(size * aug_size))
    choice2 = np.random.randint(size, size=int(size * aug_size))
    choice3 = np.random.randint(size, size=int(size * aug_size))

    new_matrices = np.concatenate(
        [
            matrices,
            matrices_rot_1[choice1, ...],
            matrices_rot_2[choice2, ...],
            matrices_rot_3[choice3, ...],
        ],
        axis=0,
    )
    new_features = np.concatenate(
        [
            features,
            features_rot_1[choice1],
            features_rot_2[choice2],
            features_rot_3[choice3],
        ],
        axis=0,
    )
    new_target = np.concatenate(
        [
            target,
            target[choice1],
            target[choice2],
            target[choice3],
        ]
    )

    return new_matrices, new_features, new_target


def build_extended_features(
    features_train: np.ndarray, features_valid: np.ndarray, features_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dir_train = to_XY_astropy(features_train[:, 4], features_train[:, 5])
    dir_valid = to_XY_astropy(features_valid[:, 4], features_valid[:, 5])
    dir_test = to_XY_astropy(features_test[:, 4], features_test[:, 5])

    extended_features_train = np.concatenate(
        [features_train, np.array(dir_train).T], axis=1
    )
    extended_features_valid = np.concatenate(
        [features_valid, np.array(dir_valid).T], axis=1
    )
    extended_features_test = np.concatenate(
        [features_test, np.array(dir_test).T], axis=1
    )

    return extended_features_train, extended_features_valid, extended_features_test


def scale_features(
    features_train: np.ndarray, features_valid: np.ndarray, features_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc_features = StandardScaler()
    sc_features.fit(features_train)

    scaled_features_train = sc_features.transform(features_train)
    scaled_features_valid = sc_features.transform(features_valid)
    scaled_features_test = sc_features.transform(features_test)

    return scaled_features_train, scaled_features_valid, scaled_features_test
