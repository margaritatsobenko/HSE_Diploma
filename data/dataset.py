import numpy as np
import torch
from torch.utils.data import Dataset


class LinearModelDataset(Dataset):
    def __init__(
        self,
        target: np.ndarray,
        features: np.ndarray,
    ):
        self.size = len(target)
        self.target = torch.tensor(target, dtype=torch.float)
        self.features = torch.tensor(features, dtype=torch.float)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return {"target": self.target[idx], "features": self.features[idx]}


class CNNModelDataset(Dataset):
    def __init__(
        self,
        target: np.ndarray,
        features: np.ndarray,
        matrices: np.ndarray,
    ):
        self.size = len(target)
        self.target = torch.tensor(target, dtype=torch.float)

        self.features = torch.tensor(features, dtype=torch.float)

        self.matrices = torch.tensor(matrices, dtype=torch.float)
        self.matrices = self.matrices.permute(0, 3, 1, 2)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return {
            "target": self.target[idx],
            "features": self.features[idx],
            "matrices": self.matrices[idx],
        }
