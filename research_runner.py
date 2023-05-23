import glob
import os
from datetime import datetime
from typing import Dict, Tuple, List

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GNNLoader

from astrophysics import get_base_loader_config
from astrophysics.losses import Loss
from astrophysics.losses.focal_loss import FocalLoss
from astrophysics.models.cnn import CNN
from astrophysics.models.gnn import GNN_1
from astrophysics.models.mlp import MLP


class ResearchRunner:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        refresh_rate: int = 20,
        log_every_n_steps: int = 25,
        log_folder: str = "tb_logs",
        checkpoint_path: str = "./model_checkpoints",
    ):
        self.model_name = model_name

        self.batch_size = batch_size
        self.log_folder = log_folder
        self.refresh_rate = refresh_rate
        self.log_every_n_steps = log_every_n_steps

        self.checkpoint_path = checkpoint_path

        self.val_loader_config = get_base_loader_config(
            batch_size=batch_size, shuffle=False
        )
        self.train_loader_config = get_base_loader_config(
            batch_size=batch_size, shuffle=True
        )

    def get_last_ckpt(self, format_: str = "ckpt") -> str:
        list_of_files = glob.glob(os.path.join(self.checkpoint_path, f"*.{format_}"))
        return max(list_of_files, key=os.path.getctime)

    def create_checkpoint_callback(
        self,
        mode: str = "max",
        monitor: str = "val_roc_auc_epoch",
        filename: str = "{epoch:02d}-{val_roc_auc_epoch:.2f}",
    ) -> Callback:
        return ModelCheckpoint(
            mode=mode,
            monitor=monitor,
            dirpath=self.checkpoint_path,
            filename=f"{str(int(datetime.timestamp(datetime.now())))}_{filename}",
        )

    def load_model(
        self,
        checkpoint_path: str,
        loss_computer: Loss,
        params: Dict[str, Dict[str, int]],
    ) -> LightningModule:
        if self.model_name == "mlp":
            return MLP.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                loss_computer=loss_computer,
                **params[self.model_name],
            )
        if self.model_name == "cnn":
            return CNN.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                loss_computer=loss_computer,
                **params[self.model_name],
            )
        if self.model_name == "gnn_1":
            return GNN_1.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                loss_computer=loss_computer,
                **params[self.model_name],
            )

    @staticmethod
    def create_trainer(
        logger: Logger,
        callbacks: List[Callback],
        max_epochs: int,
        log_every_n_steps: int,
    ) -> Trainer:
        # return Trainer(
        #     devices=4,
        #     precision=32,
        #     logger=logger,
        #     accelerator="gpu",
        #     callbacks=callbacks,
        #     max_epochs=max_epochs,
        #     log_every_n_steps=log_every_n_steps,
        #     strategy="ddp_spawn",
        # )

        return Trainer(
            devices=1,
            precision=32,
            logger=logger,
            accelerator="gpu",
            callbacks=callbacks,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            # strategy="ddp_spawn",
            profiler="simple"
        )

    def validation_step(
        self, trainer: Trainer, dataloader: DataLoader
    ) -> List[Dict[str, float]]:
        return trainer.validate(
            ckpt_path=self.get_last_ckpt(),
            dataloaders=dataloader,
        )

    def inner_run(
        self,
        model,
        train_loader: EVAL_DATALOADERS,
        val_loader: EVAL_DATALOADERS,
        num_epochs: int,
        use_focal_loss: bool,
        params,
    ) -> Tuple[float, float]:
        logger = TensorBoardLogger(self.log_folder, name=self.model_name)

        log_every_n_steps = min(
            self.log_every_n_steps, len(train_loader) // torch.cuda.device_count() + 1
        )

        tqdm_progress_bar = TQDMProgressBar(refresh_rate=self.refresh_rate)
        checkpoint_callback = self.create_checkpoint_callback()

        callbacks = [tqdm_progress_bar, checkpoint_callback]

        trainer = self.create_trainer(logger, callbacks, num_epochs, log_every_n_steps)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if use_focal_loss:
            model = self.load_model(
                self.get_last_ckpt(),
                loss_computer=FocalLoss(gamma=3),
                params=params,
            )
            trainer = self.create_trainer(logger, callbacks, 3, log_every_n_steps)
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

        train_scores = self.validation_step(trainer, train_loader)[0]
        validation_scores = self.validation_step(trainer, val_loader)[0]

        return (
            train_scores["val_roc_auc_epoch"],
            validation_scores["val_roc_auc_epoch"],
        )

    def run(
        self,
        model,
        train_dataset,
        val_dataset,
        num_epochs: int = 50,
        use_focal_loss: bool = False,
        params=None,
    ):
        if self.model_name == "gnn_1":
            train_loader = GNNLoader(
                train_dataset, num_workers=4, **self.train_loader_config
            )
            validation_loader = GNNLoader(
                val_dataset, num_workers=2, **self.val_loader_config
            )
        else:
            train_loader = DataLoader(train_dataset, **self.train_loader_config)
            validation_loader = DataLoader(val_dataset, **self.val_loader_config)

        train_score, validation_score = self.inner_run(
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            num_epochs=num_epochs,
            use_focal_loss=use_focal_loss,
            params=params,
        )

        print(f"Train ROC-AUC score: {train_score}")
        print(f"Validation ROC-AUC score: {validation_score}")
