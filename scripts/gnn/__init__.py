from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from astrophysics.data.gnn.dataset import GNNDataset
from astrophysics.models.gnn import GNN_1


@torch.no_grad()
def evaluate_gnn(
    clf: GNN_1,
    dataset_train: GNNDataset,
    dataset_valid: GNNDataset,
    dataset_test: GNNDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions_train = []

    cuda = torch.device('cuda')
    clf = clf.to(device=cuda)

    for i in tqdm(range(len(dataset_train)), desc="Evaluate Train"):
        data_batch = dataset_train[i].to(device=cuda)
        predictions_train.append(
            clf.predict(
                data_batch.x,
                data_batch.features,
                data_batch.edge_index,
                data_batch.batch,
            ).item()
        )

    predictions_valid = []

    for i in tqdm(range(len(dataset_valid)), desc="Evaluate Validation"):
        data_batch = dataset_valid[i].to(device=cuda)
        predictions_valid.append(
            clf.predict(
                data_batch.x,
                data_batch.features,
                data_batch.edge_index,
                data_batch.batch,
            ).item()
        )

    predictions_test = []

    for i in tqdm(range(len(dataset_test)), desc="Evaluate Test"):
        data_batch = dataset_test[i].to(device=cuda)
        predictions_test.append(
            clf.predict(
                data_batch.x,
                data_batch.features,
                data_batch.edge_index,
                data_batch.batch,
            ).item()
        )

    return (
        np.array(predictions_train),
        np.array(predictions_valid),
        np.array(predictions_test),
    )


@torch.no_grad()
def predict_proba_gnn(clf, dataset):
    probas = []

    cuda = torch.device('cuda')
    clf = clf.to(device=cuda)

    for i in tqdm(range(len(dataset)), desc="Predict probas"):
        data_batch = dataset[i].to(device=cuda)
        logit = clf.predict(
            data_batch.x,
            data_batch.features,
            data_batch.edge_index,
            data_batch.batch,
        )

        probas.append(torch.sigmoid(logit).item())

    return np.array(probas)
