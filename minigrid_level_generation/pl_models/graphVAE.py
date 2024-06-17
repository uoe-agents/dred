# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import logging

import torch
import pytorch_lightning as pl
import hydra

from models.dred_minigrid import GraphGCNEncoder, GraphMLPDecoder, graphVAE_elbo_pathwise

logger = logging.getLogger(__name__)


class LightningGraphVAE(pl.LightningModule):

    def __init__(self, config_model, config_optim, hparams_model, config_logging, **kwargs):
        super(LightningGraphVAE, self).__init__()
        self.save_hyperparameters()

        self.encoder = GraphGCNEncoder(self.hparams.config_model.encoder, self.hparams.config_model.shared_parameters)
        self.decoder = GraphMLPDecoder(self.hparams.config_model.decoder, self.hparams.config_model.shared_parameters)

        device = torch.device("cuda" if self.hparams.config_model.model.accelerator == "gpu" else "cpu")
        self.to(device)

    def forward(self, X):
        return self.elbo(X)

    def elbo(self, X):
        outputs = self.all_model_outputs_pathwise(X, num_samples=self.hparams.config_model.model.num_variational_samples)
        return outputs["elbos"]

    def loss(self, X):
        outputs = self.all_model_outputs_pathwise(X, num_samples=self.hparams.config_model.model.num_variational_samples)
        return outputs["loss"]

    def all_model_outputs_pathwise(self, X, num_samples: int = None):
        if num_samples is None: num_samples = self.hparams.config_model.model.num_variational_samples
        outputs = \
        graphVAE_elbo_pathwise(X, encoder=self.encoder, decoder=self.decoder,
                                 num_samples=num_samples,
                                 elbo_coeffs=self.hparams.hparams_model.loss.elbo_coeffs,
                                 output_keys=self.hparams.config_model.model.outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        X, labels = batch
        loss = self.loss(X)

        self.log('loss/train', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx, **kwargs):
        X, labels = batch
        outputs = \
            self.all_model_outputs_pathwise(X, num_samples=self.hparams.config_model.model.num_variational_samples)
        return outputs

    def predict_step(self, batch, batch_idx, **kwargs):
        dataloader_idx = 0
        return self.validation_step(batch, batch_idx, dataloader_idx, **kwargs)

    def test_step(self, batch, batch_idx, **kwargs):
        dataloader_idx = 0
        return self.validation_step(batch, batch_idx, dataloader_idx, **kwargs)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.config_optim, params=self.parameters())
        return optimizer

    def on_train_start(self):
        # Proper logging of hyperparams and metrics
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)

    def save_torch_checkpoint(self, checkpoint_path):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "hyper_parameters": {},
        }
        from omegaconf import OmegaConf
        for key in self.hparams:
            checkpoint["hyper_parameters"][key] = OmegaConf.to_container(self.hparams[key], resolve=True)
        torch.save(checkpoint, checkpoint_path)