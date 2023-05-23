import datetime
import sys

import numpy as np

sys.path.append("/workspace/code/")

from astrophysics.data import load_train_val_test
from astrophysics.data.dataset import LinearModelDataset, CNNModelDataset
from astrophysics.research_runner import ResearchRunner
from astrophysics.models.cnn import CNN


if __name__ == "__main__":
    # date_ = datetime.date(2023, 3, 4)
    # date_ = datetime.date(2023, 3, 25)
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

    dataset_train = CNNModelDataset(target_train, features_train, matrices_train)
    dataset_valid = CNNModelDataset(target_valid, features_valid, matrices_valid)

    clf = CNN(n_features=features_train.shape[1], n_mat=3)
    research_runner = ResearchRunner(
        model_name="cnn",
        batch_size=512,
    )

    research_runner.run(
        clf,
        dataset_train,
        dataset_valid,
        num_epochs=10,
        use_focal_loss=True,
        params={"cnn": {"n_features": features_train.shape[1], "n_mat": 3}},
    )
