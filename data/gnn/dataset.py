import torch
from torch_geometric.data import Data, Dataset

from astrophysics.data.gnn.utils import get_all_edges_real_dist


class GNNDataset(Dataset):
    def __init__(self, X, y, F):
        self.X = torch.tensor(X, dtype=torch.float)
        self.X = self.X.reshape(self.X.shape[0], -1, 3)

        self.y = torch.tensor(y, dtype=torch.float)
        self.F = torch.tensor(F, dtype=torch.float)

        self.edges, self.dists = get_all_edges_real_dist(16)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return Data(
            x=self.X[idx],
            edge_index=self.edges,
            edge_attr=self.dists,
            y=self.y[idx],
            features=self.F[idx],
        )
