# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import logging

import argparse
from typing import Dict
import sys
from omegaconf import DictConfig

from util import *

from .graph_metrics import shortest_paths, object_count_vs_spl, object_density, prepare_graph, compute_metrics, \
    get_non_nav_spl
from .logger_callbacks import GraphVAELogger
from .multiprocessing import Parser

logger = logging.getLogger(__name__)

def get_size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif torch.is_tensor(obj):
        size += obj.nelement() * obj.element_size()
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


def seed_everything(seed=20211201):
    """
    Helper to seed random number generators.

    Note, however, that even with the seeds fixed, some non-determinism is possible.

    For more details read <https://pytorch.org/docs/stable/notes/randomness.html>.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # The below might help to make it more deterministic
    # but will hurt the performance significantly
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def check_unique(input: torch.Tensor):
    """Returns a mask of all unique elements in the input."""
    if isinstance(input, dict):
        input = torch.stack(list(input.values()), dim=-1)
    elif not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    _, ind, counts = torch.unique(input, dim=0, return_inverse=True, return_counts=True, sorted=False)
    return counts[ind] == 1

def check_novel(batch, dataset):
    """
    Returns a mask of size B indicating which tensors in batch are not present in the dataset.

    Args:
    batch (Tensor): a batch of tensors to check novelty of dim (B, ...)
    dataset: a batch of tensors to check each batch tensor against with dim (N, ...)
    """

    if isinstance(batch, dict):
        assert isinstance(dataset, dict), "If batch is a dict, dataset must also be a dict"
        min_feature_set = list(set(dataset.keys()).intersection(set(batch.keys())))
        batch = torch.stack([batch[key] for key in min_feature_set], dim=-1)
    elif not isinstance(batch, torch.Tensor):
        batch = torch.tensor(batch)
    if isinstance(dataset, dict):
        dataset = torch.stack([dataset[key] for key in min_feature_set], dim=-1)
    elif not isinstance(dataset, torch.Tensor):
        dataset = torch.tensor(dataset)

    #Best to avoid loading dataset on GPU as it is likely to be large
    device = batch.device
    if dataset.device == torch.device("cpu"):
        batch = batch.cpu()

    dataset = torch.unique(dataset, dim=0, sorted=False)

    novel = []
    for b in batch:
        data = torch.concat((b.unsqueeze(0), dataset), dim=0)
        _, ind, counts = torch.unique(data, dim=0, return_inverse=True, return_counts=True, sorted=False)
        if counts[ind[0]] == 1:
            novel.append(True)
        else:
            novel.append(False)

    return torch.tensor(novel, dtype=torch.bool).to(device)

def latent_interpolation(model, minibatch:torch.Tensor, Z_mean=None, Z_std=None, n_interp:int = 4, dim_r_threshold=None, interpolation_scheme:str = 'linear', img_dims: Tuple[int, int, int] = None, figsize: Tuple[int, int] = (10, 10),
                              labels=None, latent_sampling=False, device='cpu', alpha=0.5, title=None):

    #assert len(minibatch) % 2 == 0

    if Z_mean is None: Z_mean = 0
    if Z_std is None: Z_std = 1

    model.eval()
    latents, logvar = model.encoder(minibatch)

    if latent_sampling:
        rand = torch.randn(latents.shape)
        latents = latents + torch.exp(0.5*logvar) * rand

    # remove dimensions with average standard deviation of 1 (e.g. uninformative)
    if dim_r_threshold is not None:
        kept_dims = torch.where(Z_std < dim_r_threshold)[0].cpu().numpy()
        print(f"Useful dimensions: {len(kept_dims)} out of {latents.shape[-1]}")
        original_latents = latents.clone()
        latents = latents[..., kept_dims]
        Z_std_red = Z_std[kept_dims]
        Z_mean_red = Z_mean[kept_dims]
    else:
        Z_std_red = Z_std.clone()
        Z_mean_red = Z_mean.clone()

    latents = (latents - Z_mean_red) / Z_std_red


    if interpolation_scheme == 'linear':
        latents_dist = torch.cdist(latents, latents, p=2)
        eps = 5e-3
    if interpolation_scheme == 'polar':
        eps = 5e-3
        latents_dist = cdist_polar(latents, latents)

    latents_dist[latents_dist<=eps] = float('inf')
    dist, pair_indices = latents_dist.min(axis=0)
    dist = dist.cpu().numpy()
    pair_indices = pair_indices.cpu().numpy()

    distances = {}
    for ind_a, ind_b in enumerate(pair_indices):
        assert ind_a != ind_b
        key = (min(ind_a, ind_b), max(ind_a, ind_b))
        distances[key] = dist[ind_a]

    # sort the distances in ascending order
    sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    interpolation_points = np.linspace(0, 1, n_interp+2)[1:-1]
    interpolated_latents_dict = {}
    for pair in sorted_distances.keys():
        interpolated_latents = [latents[pair[0]],]
        for loc in interpolation_points:
            if interpolation_scheme == 'linear':
                interp = lerp(loc, latents[pair[0]], latents[pair[1]])
            elif interpolation_scheme == 'polar':
                interp = slerp(loc, latents[pair[0]], latents[pair[1]])
            interpolated_latents.append(interp)
        interpolated_latents.append(latents[pair[1]])
        interpolated_latents_dict[pair] = interpolated_latents

    # convert back to tensor
    l_t = [torch.stack(i) for i in list(interpolated_latents_dict.values())]
    interpolated_latents_grid = torch.stack(l_t).to(device)

    # Go back to original sample
    interpolated_latents_grid = (interpolated_latents_grid * Z_std_red) + Z_mean_red

    #reassembly
    if dim_r_threshold is not None:
        interpolated_latents_grid_r = torch.zeros(*interpolated_latents_grid.shape[:-1], original_latents.shape[-1], dtype=torch.float).to(device)
        interpolated_latents_grid_r[..., kept_dims] = interpolated_latents_grid
    else:
        interpolated_latents_grid_r = interpolated_latents_grid

    # Obtain the interpolated samples
    interpolated_logits = model.decoder(interpolated_latents_grid_r)
    interpolated_samples = model.decoder.param_b(interpolated_logits)

    return interpolated_latents_grid, interpolated_samples

def cdist_polar(x1, x2, eps=1e-08):
    return torch.abs(torch.nn.functional.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1, eps=eps))

def tuple_string(s):
    s = s.replace("(", "").replace(")", "")
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        try:
            x, y = map(int, s.split(','))
            return x, y
        except:
            try:
                return (int(s),)
            except:
                raise argparse.ArgumentTypeError("Each tuple must be '(x,y,z)' or 'x,y,z', up to 3 elements")

def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

        # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
        return d

    # declaring a class
    class AttrDict:
        pass

    # constructor of the class passed to obj
    obj = AttrDict()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])

    return obj

def rgba2rgb(imgs:torch.Tensor, background=(1, 1, 1))->torch.Tensor:
    """Converts a batch of RGB images to their RGBA equivalent."""

    assert imgs.ndim == 4, "Input must be of dimensions (B, C, H, W)"

    B, ch, row, col = imgs.shape

    assert ch==4, "Input must be a batch of RGBA images, with 4 channels"

    rgb = torch.zeros((B, 3, row, col), dtype=torch.float).to(imgs.device)

    r, g, b, a = imgs[:, 0, ...], imgs[:, 1, ...], imgs[:, 2, ...], imgs[:, 3, ...]

    R, G, B = background

    rgb[:, 0, ...] = r * a + (1.0 - a) * R
    rgb[:, 1, ...] = g * a + (1.0 - a) * G
    rgb[:, 2, ...] = b * a + (1.0 - a) * B

    return rgb


def save_state(cfg, model, optimizer, file, model_states: List, optim_states: List):
    return torch.save({
        'config': cfg,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'additional_model_states': model_states,
        'additional_optimizer_states': optim_states,
    }, file)


def load_state(file, model=None, optimizer=None, model_type=None, optim_type=None, device=None, load_different_state:int = None):
    checkpoint = torch.load(file, map_location=device)
    parser = checkpoint['argparser']
    if device is not None: parser.device = device
    if model is None:
        model = model_type(parser).to(parser.device)
        optimizer = optim_type(model.parameters(), lr=parser.learning_rate)
    if load_different_state is None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        if checkpoint['additional_model_states'][load_different_state] is None or checkpoint['additional_optimizer_states'][load_different_state] is None:
            raise RuntimeError(f"Requested state dicts of index {load_different_state} not present in checkpoint file")
        model.load_state_dict(checkpoint['additional_model_states'][load_different_state])
        optimizer.load_state_dict(checkpoint['additional_optimizer_states'][load_different_state])
    return model, optimizer, parser


def assemble_extra_data(entry: Tuple[List[dgl.DGLGraph], Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    # Takes as input entry = (graphs, extra), the output of a dgl.load_graphs call (https://docs.dgl.ai/_modules/dgl/data/graph_serialize.html#load_graphs).
    # Returns a dictionary splitting the graphs into N chunks of equal size, where N is the number of keys returned in extras.
    # Looks like this: output = {"edge_graphs":{key1: [graphs_chunk1], key2: [graphs_chunk2], ...}}
    graphs, extra = entry
    num_chunks = len(extra.keys())
    len_chunks = len(graphs) // num_chunks
    chunks = [graphs[x:x + len_chunks] for x in range(0, len(graphs), len_chunks)]
    extra_data = {"edge_graphs":{}}
    for c, key in enumerate(extra.keys()):
        extra_data["edge_graphs"][key] = chunks[c]
    return extra_data


def compute_weights(scores: np.ndarray, params: DictConfig)-> np.ndarray:
    if params.distribution == 'power':
        weights = (scores.clip(0)) ** (1. / params.temperature)
    elif params.distribution == 'power_rank':
        temp = np.flip(scores.argsort())
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp)) + 1
        weights = 1 / ranks ** (1. / params.temperature)
    else:
        raise ValueError(f"Unknown distribution: {params.distribution}")

    z = np.sum(weights)
    if z > 0:
        weights /= z
    else:
        weights = np.ones_like(weights, dtype=np.float) / len(weights)
        weights /= np.sum(weights)

    return weights
