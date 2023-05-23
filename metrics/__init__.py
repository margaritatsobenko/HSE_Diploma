from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score


def bootstrap_aggregate_metric(
    metric: Callable,
    target: np.ndarray,
    predictions: np.ndarray,
    err: float = 0.05,
    iters: int = 100,
    size: float = 1.0,
) -> Tuple[float, float, float]:
    sample_size = int(size * len(predictions))

    assert sample_size > 0, "Zero size error"

    values = []
    for _ in range(iters):
        indices = np.random.choice(len(predictions), sample_size)
        values.append(metric(target[indices], predictions[indices]))

    values = sorted(values)

    gamma = int(iters * err / 2)

    return metric(target, predictions), values[gamma], values[iters - 1 - gamma]


def print_metrics_info(
    metric_value: float, lower_bound, upper_bound, err: float = 0.95
):
    gamma = round(1 - err, 3)
    print(f"Value: {metric_value}")
    print(f"Confidence interval: [{lower_bound}, {upper_bound}], gamma = {gamma}")


def evaluate_samples(
    samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
    err: float = 0.05,
):
    for data_type, (target, predictions) in samples.items():
        print(data_type)
        print(roc_auc_score.__name__)
        metric_value, lower_bound, upper_bound = bootstrap_aggregate_metric(
            roc_auc_score, target, predictions, err=err
        )

        print_metrics_info(metric_value, lower_bound, upper_bound, err=err)
        print()


def evaluate_tpr_fpr(
    target, predictions: np.ndarray, threshold: float, err: float = 0.05
):
    for metric in [metric_tpr, metric_fpr]:
        print(metric.__name__)
        metric_value, lower_bound, upper_bound = bootstrap_aggregate_metric(
            metric, target, predictions > threshold, err=err
        )

        print_metrics_info(metric_value, lower_bound, upper_bound, err=err)
        print()


def find_best_threshold(
    target,
    predictions: np.ndarray,
    global_dict,
    threshold=0.5,
    fpr_value=1e-2,
    step=0.01,
):
    if threshold in global_dict:
        return global_dict[-1]

    global_dict.append(threshold)

    fpr = metric_fpr(target, predictions > threshold)

    if fpr < fpr_value:
        return find_best_threshold(
            target, predictions, global_dict, threshold - step, fpr_value
        )
    else:
        return find_best_threshold(
            target, predictions, global_dict, threshold + step, fpr_value
        )


def metric_tpr(y_true, y_pred):
    return recall_score(y_true, y_pred)


def metric_fpr(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    return FP / (FP + TN)


def tp_fp_ratio(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    return TP / (FP**0.5 + 1e-8)


def plot_proba(probas: np.ndarray, labels: np.ndarray, bins: int = 50, name: str = ""):
    plt.style.use("seaborn")

    labels_unique = np.unique(labels)

    sep_by_class = [probas[labels == label] for label in labels_unique]

    labels_dict = {0: "Протон", 1: "Гамма-квант"}
    label_names = [labels_dict[x] for x in labels_unique]

    colors = ["red", "lime"]
    labels_str = list(map(str, label_names))

    plt.figure(dpi=512)

    plt.hist(
        x=sep_by_class[0],
        bins=bins,
        label=labels_str[0],
        stacked=True,
        alpha=0.8,
        color=colors[0],
    )
    plt.hist(
        x=sep_by_class[1],
        bins=bins,
        label=labels_str[1],
        stacked=True,
        alpha=0.8,
        color=colors[1],
    )

    plt.legend(loc="best")

    plt.xlabel("Вероятность того, что частица является гамма-квантом")
    plt.ylabel("Количество частиц")

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(f"/workspace/data/astropy_features/images/{name}")
    plt.show()
