# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import os
import torch
from dgl.dataloading import GraphDataLoader

from models import GraphVAE
from util import interpolate_between_pairs, Nav2DTransforms, DotDict


class LevelGenerator():

    def __init__(self,
                 device,
                 level_info,
                 model_checkpoint_path,
                 interpolations_per_pair=6,
                 interpolation_scheme='polar',
                 fixed_point_interpolation=False,
                 include_endpoints_in_interpolation=True,
                 max_batch_size=0):

        self.device = device
        self.level_info = level_info
        self.interpolations_per_pair = interpolations_per_pair
        self.interpolation_scheme = interpolation_scheme
        self.fixed_point_interpolation = fixed_point_interpolation
        self.include_endpoints_in_interpolation = include_endpoints_in_interpolation
        self.max_batch_size = max_batch_size # to avoid memory issues on smaller GPUs

        model_checkpoint_path = os.path.expandvars(os.path.expanduser(model_checkpoint_path))
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        self.model = GraphVAE(**DotDict(checkpoint["hyper_parameters"]))
        self.model.load_from_checkpoint(checkpoint)
        self.model.eval()

    def generate_levels(self, base_latent_params_batch, pair_ids, base_seeds=None, base_labels=None):
        interp_params_batch, closest_parent_ids = \
            self._interpolate_latent_distribution_parameters(base_latent_params_batch, pair_ids)
        mean, std = interp_params_batch
        Z = mean + std * torch.randn_like(std)
        interp_levels = self._decode_levels(Z)
        if base_seeds is not None:
            parent_seeds = []
            for pair in pair_ids:
                parent_seeds.extend([tuple([base_seeds[pair[0]], base_seeds[pair[1]]])]*self.interpolations_per_pair)
        else:
            parent_seeds = None
        if base_labels is not None:
            labels = [base_labels[idx] for idx in closest_parent_ids]
        else:
            labels = None
        return interp_levels, interp_params_batch, parent_seeds, labels

    def compute_latent_distribution_parameters(self, dataset_sampler, split='train'):
        dataset = dataset_sampler.dataset_train if split == 'train' else dataset_sampler.dataset_test
        mbs = len(dataset) if self.max_batch_size == 0 else self.max_batch_size
        latent_params = []
        data_loader = GraphDataLoader(dataset=dataset, batch_size=mbs, shuffle=False)
        for (mbatch, labels) in data_loader:
            mean, std = self.model.encoder(mbatch.to(self.device))
            params = [(m, s) for m, s in zip(mean.cpu().detach(), std.cpu().detach())]
            latent_params.extend(params)
        return latent_params

    def _interpolate_latent_distribution_parameters(self,
                                                    base_latent_params_batch,
                                                    pair_ids):

        mean, std = (base_latent_params_batch[0].to(self.device), base_latent_params_batch[1].to(self.device))
        interp_mean = interpolate_between_pairs(pair_ids,
                                                mean,
                                                self.interpolations_per_pair,
                                                self.interpolation_scheme,
                                                self.include_endpoints_in_interpolation)
        if std is not None and not self.fixed_point_interpolation:
            interp_std = interpolate_between_pairs(pair_ids,
                                                    std,
                                                    self.interpolations_per_pair,
                                                    'linear',
                                                    self.include_endpoints_in_interpolation)
        else:
            interp_std = torch.zeros_like(interp_mean)
        interp_params_batch = (interp_mean, interp_std)
        closest_parent_ids = []
        for pair in pair_ids:
            closest_parents_ = [pair[0]] * (self.interpolations_per_pair // 2) + \
                               [pair[1]] * (self.interpolations_per_pair // 2)
            closest_parent_ids.extend(closest_parents_)

        return interp_params_batch, closest_parent_ids

    def _decode_levels(self, level_embeddings):
        logits = self.model.decoder(level_embeddings)
        g_features = self.model.decoder.to_graph_features(logits)
        # specify a color_config argument if using non default colors for objects in the levels
        levels = Nav2DTransforms.graph_features_to_minigrid(g_features, self.level_info)
        return levels