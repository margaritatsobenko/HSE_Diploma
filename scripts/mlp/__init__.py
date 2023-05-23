import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_mlp(clf, dataset_train, dataset_valid, dataset_test):
    predictions_train = []

    for i in tqdm(range(len(dataset_train)), desc="Evaluate Train"):
        x = dataset_train[i]
        predictions_train.append(clf.model(x["features"]).item())

    predictions_valid = []

    for i in tqdm(range(len(dataset_valid)), desc="Evaluate Validation"):
        x = dataset_valid[i]
        predictions_valid.append(clf.model(x["features"]).item())

    predictions_test = []

    for i in tqdm(range(len(dataset_test)), desc="Evaluate Test"):
        x = dataset_test[i]
        predictions_test.append(clf.model(x["features"]).item())

    return (
        np.array(predictions_train),
        np.array(predictions_valid),
        np.array(predictions_test),
    )


@torch.no_grad()
def predict_proba_mlp(clf, dataset):
    probas = []

    for i in tqdm(range(len(dataset)), desc="Predict probas"):
        x = dataset[i]
        logit = clf.model(x["features"])

        probas.append(torch.sigmoid(logit).item())

    return np.array(probas)
