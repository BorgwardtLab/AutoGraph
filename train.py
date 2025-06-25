import os
import logging
import hydra
from pyprojroot import here
import numpy as np
from pathlib import Path

import graph_tool
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from autograph.models.seq_models import SequenceModel


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver('eval', eval)

log = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3", config_path=str(here() / "configs"), config_name="train"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.model.pretrained_path is None:
        model = SequenceModel(cfg)
    else:
        log.info(f"Loading model from {cfg.model.pretrained_path}...")
        model = SequenceModel.load_from_checkpoint(cfg.model.pretrained_path, model=cfg.model)
        os.symlink(
            Path(cfg.model.pretrained_path).resolve(),
            Path(cfg.logs.path) / "pretrained.ckpt",
        )
        model.update_cfg(cfg)
    datamodule = model._datamodule

    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(project="AutoGraph", config=OmegaConf.to_container(cfg, resolve=True))
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    model_ckpt_cls = pl.callbacks.ModelCheckpoint

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        model_ckpt_cls(
            monitor=f'val/{datamodule.val_metric[0]}',
            dirpath=cfg.logs.path,
            filename=cfg.model.model_name,
            mode=f'{datamodule.val_metric[1]}',
        )
    ]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule)

    trainer.save_checkpoint(f"{cfg.logs.path}/{cfg.model.model_name}-last.ckpt")

    model.cfg.sampling.num_samples = -1 # balanced number
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
