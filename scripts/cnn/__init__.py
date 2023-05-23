from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from astrophysics.data.dataset import CNNModelDataset
from astrophysics.models.cnn import CNN


@torch.no_grad()
def evaluate_cnn(
    clf: CNN,
    dataset_train: CNNModelDataset,
    dataset_valid: CNNModelDataset,
    dataset_test: CNNModelDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions_train = []

    for i in tqdm(range(len(dataset_train)), desc="Evaluate Train"):
        x = dataset_train[i]
        predictions_train.append(
            clf.predict(
                x["features"].view(1, -1), x["matrices"].view(1, -1, 16, 16)
            ).item()
        )

    predictions_valid = []

    for i in tqdm(range(len(dataset_valid)), desc="Evaluate Validation"):
        x = dataset_valid[i]
        predictions_valid.append(
            clf.predict(
                x["features"].view(1, -1), x["matrices"].view(1, -1, 16, 16)
            ).item()
        )

    predictions_test = []

    for i in tqdm(range(len(dataset_test)), desc="Evaluate Test"):
        x = dataset_test[i]
        predictions_test.append(
            clf.predict(
                x["features"].view(1, -1), x["matrices"].view(1, -1, 16, 16)
            ).item()
        )

    return (
        np.array(predictions_train),
        np.array(predictions_valid),
        np.array(predictions_test),
    )


@torch.no_grad()
def predict_proba_cnn(clf, dataset):
    probas = []

    for i in tqdm(range(len(dataset)), desc="Predict probas"):
        x = dataset[i]
        logit = clf.predict(x["features"].view(1, -1), x["matrices"].view(1, -1, 16, 16))

        probas.append(torch.sigmoid(logit).item())

    return np.array(probas)
