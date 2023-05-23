import torch.nn as nn

from astrophysics.models import BaseModel
from astrophysics.models import OPTIMIZER_CONFIG_01


class MLP(BaseModel):
    def __init__(self, input_features: int, loss_computer=None):
        super().__init__(optimizer_config=OPTIMIZER_CONFIG_01, loss_computer=loss_computer)
        self.model = nn.Sequential(
            nn.Linear(input_features, 2 * input_features),
            nn.LeakyReLU(),
            nn.Linear(2 * input_features, input_features),
            nn.LeakyReLU(),
            nn.Linear(input_features, input_features // 2),
            nn.LeakyReLU(),
            nn.Linear(input_features // 2, 1),
        )

    def forward_step(self, x):
        logits = self.model(x["features"])
        return {
            "loss": self.loss_computer(logits.view(-1), x["target"].float()),
            "roc_auc": self.auroc_metric(logits, x["target"]),
        }
