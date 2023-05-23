import datetime
import sys

sys.path.append("/workspace/code/")

from astrophysics.data import load_train_val_test
from astrophysics.data.dataset import CNNModelDataset
from astrophysics.research_runner import ResearchRunner
from astrophysics.models.cnn import CNN

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

    two_matrices_train = matrices_train[:, :, :, 1:]
    two_matrices_valid = matrices_valid[:, :, :, 1:]

    dataset_train = CNNModelDataset(target_train, features_train, two_matrices_train)
    dataset_valid = CNNModelDataset(target_valid, features_valid, two_matrices_valid)

    clf = CNN(n_features=features_train.shape[1], n_mat=2)
    research_runner = ResearchRunner(
        model_name="cnn",
        batch_size=512,
    )

    research_runner.run(clf, dataset_train, dataset_valid)
