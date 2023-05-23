from abc import ABC
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics.classification import BinaryAUROC

Batch = Dict[str, Optional[Tensor]]
StepOutput = Dict[str, Tensor]
OptimizerConfig = Optional[Dict[str, Any]]


OPTIMIZER_CONFIG_01 = {"name": "Adam", "parameters": {"lr": 1e-2}}
OPTIMIZER_CONFIG_02 = {"name": "RAdam", "parameters": {"lr": 1e-3}}
OPTIMIZER_CONFIG_03 = {"name": "RAdam", "parameters": {"lr": 1e-4, "weight_decay": 1}}


class BaseModel(LightningModule, ABC):
    __optimizers = {"Adam": torch.optim.Adam, "RAdam": torch.optim.RAdam}

    def __init__(
        self,
        optimizer_config: OptimizerConfig = None,
        loss_computer=None,
    ):
        super(BaseModel, self).__init__()

        self.optim_config = optimizer_config or OPTIMIZER_CONFIG_02
        # self.optim_config = optim_config or OPTIMIZER_CONFIG_03

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.auroc_metric = BinaryAUROC()

        self.loss_computer = nn.BCEWithLogitsLoss()
        if loss_computer is not None:
            self.loss_computer = loss_computer

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.__optimizers[self.optim_config["name"]](
            self.parameters(), **self.optim_config["parameters"]
        )

    def log_step_output(self, step_output: StepOutput, step_prefix: str):
        for field, tensor in step_output.items():
            self.log(
                f"{step_prefix}_{field}",
                tensor,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                # sync_dist=True,
                batch_size=256,
            )

    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        loss_output = self.forward_step(batch)
        self.log_step_output(loss_output, "train")
        return {"loss": loss_output["loss"]}

    def validation_step(self, batch: Batch, batch_idx: int):
        self.log_step_output(self.forward_step(batch), "val")

    def test_step(self, batch: Batch, batch_idx: int):
        self.log_step_output(self.forward_step(batch), "test")
