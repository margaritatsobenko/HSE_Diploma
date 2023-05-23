import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from astrophysics.models import BaseModel, OPTIMIZER_CONFIG_01


class GCN(BaseModel):
    def __init__(
        self, num_node_features: int = 2, hidden_channels: int = 64, out_dim: int = 2
    ):
        super().__init__(optimizer_config=OPTIMIZER_CONFIG_01)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, out_dim),  # num_classes
        )

    def forward_step(self, data_batch):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        logits = self.fc_layers(x)

        return {
            "loss": self.bce_loss(logits.view(-1), data_batch.y.float()),
            "roc_auc": self.auroc_metric(logits, data_batch.y),
        }


class GNN_1(BaseModel):
    def __init__(
        self,
        features_dim: int,
        num_node_features: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.features_dim = features_dim
        # Тут в конец добавим дополнительные фичи
        input_dim = hidden_channels + features_dim
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),  # num_classes
        )

    def forward_step(self, data_batch):
        x, f, edge_index, batch = (
            data_batch.x,
            data_batch.features,
            data_batch.edge_index,
            data_batch.batch,
        )

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        f = f.reshape(-1, self.features_dim)
        x = torch.cat((x, f), 1).float()
        x = self.fc_layers(x)

        return {
            "loss": self.bce_loss(x.view(-1), data_batch.y.float()),
            "roc_auc": self.auroc_metric(x, data_batch.y),
        }

    def predict(self, x, f, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        f = f.reshape(-1, self.features_dim)
        x = torch.cat((x, f), 1).float()
        x = self.fc_layers(x)
        return x
