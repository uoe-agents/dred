# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import os.path

import hydra
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl

from minigrid_level_generation.data_loaders import MinigridDataModule
from minigrid_level_generation.pl_models.graphVAE import LightningGraphVAE
from minigrid_level_generation.extra_util import *

@hydra.main(version_base=None, config_path="minigrid_level_generation/conf", config_name="config.yaml")
def run_experiment(cfg: DictConfig) -> None:

    logger = logging.getLogger(__name__)
    logger.info("Working directory : {}".format(os.getcwd()))
    process_cfg(cfg)
    seed_everything(cfg.seed)
    run_post_training_eval = cfg.get("run_post_training_eval", False)

    # Initialize logging (you will need to implement your own logging callback)
    logging_callbacks = None
    exp_logger = False
    if cfg.get("logging", False):
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        logging_callbacks = [
            GraphVAELogger(), # initialize your logging callback(s) here
        ]
        exp_logger = WandbLogger(project="abc", entity="xyz") # connect to wandb here

    model = LightningGraphVAE(config=cfg.model,
                                    config_model=cfg.model.configuration,
                                    config_optim=cfg.optim,
                                    hparams_model=cfg.model.hyperparameters,
                                    config_logging =cfg.logger)

    dataset_full_dir, cfg.dataset.path = get_dataset_dir(cfg.dataset)
    data_module = MinigridDataModule(dataset_full_dir,
                                     batch_size=cfg.dataset.batch_size,
                                     transforms=cfg.dataset.transforms,
                                     num_workers=cfg.num_cpus,
                                     val_data=cfg.dataset.val_data,
                                     no_images=cfg.dataset.no_images,
                                     held_out_tasks=cfg.dataset.held_out_tasks)

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    data_module.setup()

    trainer = pl.Trainer(accelerator=cfg.accelerator, devices=cfg.num_devices, max_epochs=cfg.epochs,
                         logger=exp_logger, callbacks=logging_callbacks)
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=[data_module.val_dataloader(), data_module.predict_dataloader()])    # run prediction (latent space viz and interpolation) in inference mode
    model_output_path = get_torch_checkpoint_path(cfg)
    model.save_torch_checkpoint(model_output_path)
    if run_post_training_eval:
        trainer.predict(dataloaders=data_module.predict_dataloader())
    # evaluate the model on a test set
    trainer.test(datamodule=data_module, ckpt_path=None)  # uses last-saved model

    if cfg.get("logging", False):
        logger.info("Terminating wandb...")
        wandb.finish()

    logger.info("Done")

def get_dataset_dir(cfg):
    base_dir = str(Path(__file__).resolve().parent)
    datasets_dir = base_dir + '/datasets/'
    data_directory = cfg.name
    data_full_dir = datasets_dir + data_directory
    return data_full_dir, data_directory

def get_torch_checkpoint_path(cfg):
    base_dir = str(Path(__file__).resolve().parent)
    checkpoint_path = os.path.join(base_dir, cfg.torch_checkpoint_dir, cfg.run_name + '.pt')
    return checkpoint_path

def process_cfg(cfg):
    # sets Auto arguments
    if cfg.dataset.data_type =='graph':
        if cfg.dataset.encoding == 'dense':
            gw_data_dim = cfg.dataset.gridworld_data_dim
            f = lambda x: (x - 2)
            cfg.dataset.max_nodes = int(f(gw_data_dim[1]) * f(gw_data_dim[2]))
        else:
            raise NotImplementedError(f"Encoding {cfg.dataset.encoding} not implemented "
                                      f"for data_type {cfg.dataset.data_type}.")


if __name__ == "__main__":
    run_experiment()
