from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")


def print_data_shapes(
    matrices: np.ndarray, features: np.ndarray, true_features: np.ndarray
):
    print(f"Shape of matrices: {str(matrices.shape).rjust(30, ' ')}")
    print(f"Shape of features: {str(features.shape).rjust(30, ' ')}")
    print(f"Shape of true features: {str(true_features.shape).rjust(25, ' ')}")


def plot_matrices_sum(matrices: np.ndarray):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(21, 7), dpi=256)
    sns.heatmap(matrices[..., 0].sum(0), square=True, ax=ax0)
    sns.heatmap(matrices[..., 1].sum(0), square=True, ax=ax1)
    sns.heatmap(matrices[..., 2].sum(0), square=True, ax=ax2)
    ax0.set_title("Arrival times (sum)")
    ax1.set_title("$e/\gamma$ detector (sum)")
    ax2.set_title("$\mu$ detector (sum)")
    plt.show()


def plot_matrices_mean(matrices: np.ndarray):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(21, 7), dpi=256)
    sns.heatmap(matrices[..., 0].mean(0), square=True, ax=ax0)
    sns.heatmap(matrices[..., 1].mean(0), square=True, ax=ax1)
    sns.heatmap(matrices[..., 2].mean(0), square=True, ax=ax2)
    ax0.set_title("Arrival times (mean)")
    ax1.set_title("$e/\gamma$ detector (mean)")
    ax2.set_title("$\mu$ detector (mean)")
    plt.show()


def show_data_imbalance(target: np.ndarray):
    """
    Предполагаем, что лейблы уже изменены, то есть
        proton == 0, gamma == 1
    :param target: является ли частица протоном (0) или гамма-квантом (1)
    :return:
    """
    stat = np.unique(target, return_counts=True)
    pr_count = stat[1][0]
    gm_count = stat[1][1]

    total_count = pr_count + gm_count

    assert len(stat[0] == 2), "Unexpected count of unique labels"

    pr_ratio = 100 * pr_count / total_count
    gm_ratio = 100 * gm_count / total_count

    print(f"Число протонов: {str(stat[1][0]).rjust(10, ' ')}, {round(pr_ratio, 2)}%")
    print(
        f"Число гамма-квантов: {str(stat[1][1]).rjust(5, ' ')}, {round(gm_ratio, 2)}%"
    )


def collect_interesting_events(features: np.ndarray, total_count: int = 5) -> List[int]:
    interesting_events = []

    for i in range(len(features)):
        if (
            (features[i][0] == 1)
            and (15 <= features[i][1] <= 16)
            and (14.5 < features[i][5] < 16.5)
            and (15 < features[i][4] < 20)
        ):
            interesting_events.append(i)
            if len(interesting_events) == total_count:
                return interesting_events

    return interesting_events


def get_vector_representation(vector: np.ndarray) -> str:
    return f"[{', '.join(list(map(str, list(vector))))}]"


def plot_event_heatmap(
    matrices: np.ndarray,
    features: np.ndarray,
    true_features: np.ndarray,
    event_num: int,
    cmap: Optional[str] = None,
):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(21, 7), dpi=256)

    # cmap "gnuplot2" for gamma
    sns.heatmap(matrices[event_num, ..., 0], square=True, ax=ax0, cmap=cmap)
    sns.heatmap(matrices[event_num, ..., 1], square=True, ax=ax1, cmap=cmap)
    sns.heatmap(matrices[event_num, ..., 2], square=True, ax=ax2, cmap=cmap)

    ax0.set_title(f"Arrival times (event #{event_num})")
    ax1.set_title(f"$e/\gamma$ detector (event #{event_num})")
    ax2.set_title(f"$\mu$ detector (event #{event_num})")

    plt.show()

    features_vector = get_vector_representation(features[event_num])
    true_features_vector = get_vector_representation(true_features[event_num])

    print(f"feature vector of event_num = {event_num} is\n\t{features_vector}")
    print(
        f"true_feature vector of event_num = {event_num} is\n\t{true_features_vector}"
    )


def plot_ze_az_angles(ze: np.ndarray, az: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=256)
    sns.histplot(ze, ax=ax1)
    sns.histplot(az, ax=ax2)
    ax1.set_title("Zenith Angle")
    ax2.set_title("Azimuth Angle")
    plt.show()


def plot_two_histograms(
    data_1: np.ndarray, data_2: np.ndarray, title_1: str, title_2: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=256)
    sns.histplot(data_1, ax=ax1)
    sns.histplot(data_2, ax=ax2)
    ax1.set_title(title_1)
    ax2.set_title(title_2)
    plt.show()
