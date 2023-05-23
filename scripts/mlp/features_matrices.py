import datetime
import sys

import numpy as np

sys.path.append("/workspace/code/")

from astrophysics.data import load_train_val_test
from astrophysics.data.dataset import LinearModelDataset
from astrophysics.research_runner import ResearchRunner
from astrophysics.models.mlp import MLP


if __name__ == "__main__":
    date_ = datetime.date(2023, 4, 9)

    (
        matrices_train,
        matrices_valid,
        matrices_test,
        features_train,
        features_valid,
        features_test,
        target_train,
        target_valid,
        target_test,
    ) = load_train_val_test(date_)

    explode_matrices_train = matrices_train.reshape(matrices_train.shape[0], -1)
    explode_matrices_valid = matrices_valid.reshape(matrices_valid.shape[0], -1)

    combined_features_train = np.concatenate(
        (features_train, explode_matrices_train), axis=-1
    )
    combined_features_valid = np.concatenate(
        (features_valid, explode_matrices_valid), axis=-1
    )

    dataset_train = LinearModelDataset(target_train, combined_features_train)
    dataset_valid = LinearModelDataset(target_valid, combined_features_valid)

    clf = MLP(combined_features_train.shape[1])
    research_runner = ResearchRunner(
        model_name="mlp",
        batch_size=512,
    )

    research_runner.run(
        clf,
        dataset_train,
        dataset_valid,
        num_epochs=5,
        use_focal_loss=True,
        params={"mlp": {"input_features": combined_features_train.shape[1]}},
    )
