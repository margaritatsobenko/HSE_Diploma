import datetime
import os
import pickle

npz_dir = "/workspace/data/npz"
base_path = f"/workspace/data/astropy_features/"

combined_dataset = {
    "matrices": "combined_gm_pr_matrices.npz",
    "features": "combined_gm_pr_features.npz",
    "true_features": "combined_gm_pr_true_features.npz",
}


def check_data_in_directory():
    print(f'\nFiles in "{npz_dir}" directory:\n\t{os.listdir(npz_dir)}')


def load_train_val_test(date_: datetime.date):
    path = os.path.join(base_path, date_.strftime("dt=%Y%m%d"))

    with open(os.path.join(path, "matrices_train.pkl"), "rb") as f:
        matrices_train = pickle.load(f)

    with open(os.path.join(path, "matrices_valid.pkl"), "rb") as f:
        matrices_valid = pickle.load(f)

    with open(os.path.join(path, "matrices_test.pkl"), "rb") as f:
        matrices_test = pickle.load(f)

    with open(os.path.join(path, "scaled_extended_features_train.pkl"), "rb") as f:
        features_train = pickle.load(f)

    with open(os.path.join(path, "scaled_extended_features_valid.pkl"), "rb") as f:
        features_valid = pickle.load(f)

    with open(os.path.join(path, "scaled_extended_features_test.pkl"), "rb") as f:
        features_test = pickle.load(f)

    with open(os.path.join(path, "target_train.pkl"), "rb") as f:
        target_train = pickle.load(f)

    with open(os.path.join(path, "target_valid.pkl"), "rb") as f:
        target_valid = pickle.load(f)

    with open(os.path.join(path, "target_test.pkl"), "rb") as f:
        target_test = pickle.load(f)

    return (
        matrices_train,
        matrices_valid,
        matrices_test,
        features_train,
        features_valid,
        features_test,
        target_train,
        target_valid,
        target_test,
    )


def load_test_data(date_: datetime.date):
    path = os.path.join(base_path, date_.strftime("dt=%Y%m%d"))

    with open(os.path.join(path, "ze_0_30_e_0_15.pkl"), "rb") as f:
        ze_0_30_e_0_15 = pickle.load(f)

    with open(os.path.join(path, "ze_0_30_e_15_16.pkl"), "rb") as f:
        ze_0_30_e_15_16 = pickle.load(f)

    with open(os.path.join(path, "ze_0_30_e_16.pkl"), "rb") as f:
        ze_0_30_e_16 = pickle.load(f)

    with open(os.path.join(path, "ze_30_60_e_0_15.pkl"), "rb") as f:
        ze_30_60_e_0_15 = pickle.load(f)

    with open(os.path.join(path, "ze_30_60_e_15_16.pkl"), "rb") as f:
        ze_30_60_e_15_16 = pickle.load(f)

    with open(os.path.join(path, "ze_30_60_e_16.pkl"), "rb") as f:
        ze_30_60_e_16 = pickle.load(f)

    return (
        ze_0_30_e_0_15,
        ze_0_30_e_15_16,
        ze_0_30_e_16,
        ze_30_60_e_0_15,
        ze_30_60_e_15_16,
        ze_30_60_e_16,
    )


def load_test_data_20_40(date_: datetime.date):
    path = os.path.join(base_path, date_.strftime("dt=%Y%m%d"))

    with open(os.path.join(path, "ze_0_20_e_15_16.pkl"), "rb") as f:
        ze_0_20_e_15_16 = pickle.load(f)

    with open(os.path.join(path, "ze_0_20_e_16.pkl"), "rb") as f:
        ze_0_20_e_16 = pickle.load(f)

    with open(os.path.join(path, "ze_20_40_e_15_16.pkl"), "rb") as f:
        ze_20_40_e_15_16 = pickle.load(f)

    with open(os.path.join(path, "ze_20_40_e_16.pkl"), "rb") as f:
        ze_20_40_e_16 = pickle.load(f)

    return (
        ze_0_20_e_15_16,
        ze_0_20_e_16,
        ze_20_40_e_15_16,
        ze_20_40_e_16,
    )
