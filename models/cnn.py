import torch
import torch.nn as nn

from astrophysics.models import BaseModel


def conv_relu(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel),
        nn.ReLU(),
    )


class CNN(BaseModel):
    def __init__(self, n_features: int, n_mat: int, n_class=1, loss_computer=None):
        super().__init__(loss_computer=loss_computer)

        self.conv1 = conv_relu(n_mat, 32, 3)
        self.conv2 = conv_relu(32, 64, 3)
        self.conv3 = conv_relu(64, 32, 3)
        self.fc1 = nn.Linear(32 * 10 * 10 + n_features, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, n_class)

        self.relu = nn.ReLU()

        self.weights_init()

    def forward_step(self, batch):
        x_features, x_mat = batch["features"], batch["matrices"]
        s0 = x_mat.shape[0]
        x_mat = self.conv1(x_mat)
        x_mat = self.conv2(x_mat)
        x_mat = self.conv3(x_mat)
        x_mat = x_mat.view(s0, -1)
        x = torch.cat((x_features, x_mat), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)

        return {
            "loss": self.loss_computer(logits.view(-1), batch["target"].float()),
            "roc_auc": self.auroc_metric(logits, batch["target"]),
        }

    def predict(self, x_features, x_mat):
        s0 = x_mat.shape[0]
        x_mat = self.conv1(x_mat)
        x_mat = self.conv2(x_mat)
        x_mat = self.conv3(x_mat)
        x_mat = x_mat.view(s0, -1)
        x = torch.cat((x_features, x_mat), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
