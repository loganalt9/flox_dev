from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from flox.flock.states import FloxWorkerState
from flox.nn import FloxModule
from flox.nn.logger.csv import CSVLogger
from flox.strategies import Strategy


class Trainer:
    def __init__(
        self,
        logger: str = "csv",
        device="cpu",
        config: dict[str, Any] | None = None,
    ):
        self.device = device
        self.config = config  # TODO: Not implemented to do anything at the moment.
        if logger == "csv":
            self.logger = CSVLogger()
        else:
            raise ValueError("Illegal value for `logger`.")

    def fit(
        self,
        model: FloxModule,
        train_dataloader: DataLoader,
        num_epochs: int,
        strategy: Strategy | None = None,
        node_state: FloxWorkerState | None = None,
        valid_dataloader: DataLoader | None = None,
        valid_ckpt_path: Path | str | None = None,
    ) -> pd.DataFrame:
        model.train()
        optimizer = model.configure_optimizers()
        self.logger.clear()

        with torch.set_grad_enabled(True):
            for epoch in range(num_epochs):
                for batch_idx, batch in enumerate(train_dataloader):
                    loss = model.training_step(batch, batch_idx)
                    optimizer.zero_grad()
                    loss.backward()

                    try:
                        strategy.wrk_on_after_train_step(node_state, loss)
                    except NotImplementedError:
                        """
                        The current strategy does not override the `wrk_on_after_train_step()` callback.
                        """
                        pass

                    optimizer.step()

                    # log data about training
                    self.logger.log_dict(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": epoch,
                            "train/batch_idx": batch_idx,
                            "train/time": datetime.datetime.now(),
                        }
                    )

                    # If a validation ``Dataset`` has been provided (i.e., the users
                    # have specified an object instance for it), then run validation.
                    if valid_dataloader is not None:
                        self.validate(model, valid_dataloader, epoch, valid_ckpt_path)

        return self.logger.to_pandas()

    def test(
        self,
        model: FloxModule,
        test_dataloader: DataLoader,
        ckpt_path: Path | str | None = None,
    ):
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                model.test_step(batch, i)

    def validate(
        self,
        model: FloxModule,
        valid_dataloader: DataLoader,
        epoch: int,
        ckpt_path: Path | str | None = None,
    ):
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                loss = model.validation_step(batch, batch_idx)
                self.logger.log_dict(
                    {
                        "valid/loss": loss.item(),
                        "valid/epoch": epoch,
                        "valid/batch_idx": batch_idx,
                        "valid/time": datetime.datetime.now(),
                    }
                )
