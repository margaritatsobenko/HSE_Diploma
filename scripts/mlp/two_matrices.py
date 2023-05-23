import datetime
import sys

sys.path.append("/workspace/code/")

from astrophysics.data import load_train_val_test
from astrophysics.data.dataset import LinearModelDataset
from astrophysics.research_runner import ResearchRunner
from astrophysics.models.mlp import MLP

if __name__ == "__main__":
    # date_ = datetime.date(2023, 3, 4)
    date_ = datetime.date(2023, 3, 25)

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

    explode_two_matrices_train = matrices_train[:, :, :, 1:].reshape(
        matrices_train.shape[0], -1
    )
    explode_two_matrices_valid = matrices_valid[:, :, :, 1:].reshape(
        matrices_valid.shape[0], -1
    )

    dataset_train = LinearModelDataset(target_train, explode_two_matrices_train)
    dataset_valid = LinearModelDataset(target_valid, explode_two_matrices_valid)

    clf = MLP(explode_two_matrices_train.shape[1])
    research_runner = ResearchRunner(
        model_name="mlp",
        batch_size=512,
    )

    research_runner.run(clf, dataset_train, dataset_valid)
