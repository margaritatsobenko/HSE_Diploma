import datetime
import sys

sys.path.append("/workspace/code/")

from astrophysics.data import load_train_val_test
from astrophysics.data.gnn.dataset import GNNDataset
from astrophysics.research_runner import ResearchRunner
from astrophysics.models.gnn import GNN_1


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

    # two_matrices_train = matrices_train[:, :, :, 1:]
    # two_matrices_valid = matrices_valid[:, :, :, 1:]
    #
    # dataset_train = GNNDataset(two_matrices_train, target_train, features_train)
    # dataset_valid = GNNDataset(two_matrices_valid, target_valid, features_valid)

    dataset_train = GNNDataset(matrices_train, target_train, features_train)
    dataset_valid = GNNDataset(matrices_valid, target_valid, features_valid)

    clf = GNN_1(hidden_channels=32, features_dim=features_train.shape[1])
    research_runner = ResearchRunner(
        model_name="gnn_1",
        batch_size=256,
    )

    research_runner.run(
        clf,
        dataset_train,
        dataset_valid,
        num_epochs=5,
        use_focal_loss=True,
        params={
            "gnn_1": {"hidden_channels": 32, "features_dim": features_train.shape[1]}
        },
    )
