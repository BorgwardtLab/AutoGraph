import os
import logging
import hydra
from pyprojroot import here
import numpy as np

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from autograph.models.seq_models import SequenceModel

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver('eval', eval)

log = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3", config_path=str(here() / "configs"), config_name="test"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Loading model from {cfg.model.pretrained_path}...")
    model = SequenceModel.load_from_checkpoint(cfg.model.pretrained_path)
    model.update_cfg(cfg)

    datamodule = model._datamodule

    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(project="GenWalk")
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
