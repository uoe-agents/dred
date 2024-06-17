# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# Following modifications by Samuel Garcin:
# - implemented make_level_generator_args(), get_node_features(), dgl_to_nx(), nx_to_dgl((), interpolate_between_pairs(), lerp(), slerp()
# - support for mixed sampling strategies in make_plr_args(), minor improvements to DotDict(), save_images()

import glob
import os
import shutil
import collections
import timeit
import random
import math
from typing import Union, List, Tuple, Any

import dgl
import networkx as nx
import numpy as np
import torch
from torchvision import utils as vutils
from torchvision.transforms import transforms as vtransforms

from envs.registration import make as gym_make
from .make_agent import make_agent
from .filewriter import FileWriter
from .transforms import *
from envs.wrappers import ParallelAdversarialVecEnv, VecMonitor, VecNormalize, \
    VecPreprocessImageWrapper, VecFrameStack, MultiGridFullyObsWrapper, CarRacingWrapper, TimeLimit
from envs.multigrid.multigrid import Grid


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __deepcopy__ = None

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    def to_dict(self):
        d = {}
        for key, value in self.items():
            if key == '__dict__':
                continue
            if isinstance(value, DotDict):
                d[key] = value.to_dict()
            else:
                d[key] = value

        return d


def array_to_csv(a):
    return ','.join([str(v) for v in a])


def cprint(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def safe_checkpoint(state_dict, path, index=None, archive_interval=None):
    filename, ext = os.path.splitext(path)
    path_tmp = f'{filename}_tmp{ext}'
    torch.save(state_dict, path_tmp)

    os.replace(path_tmp, path)

    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = f'{filename}_{index}{ext}'
            shutil.copy(path, archive_path)


def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_images(images, path=None, normalize=False, channels_first=False, nrow=None):
    if path is None:
        return

    if isinstance(images, (list, tuple)):
        images = torch.tensor(np.stack(images), dtype=torch.float)
    elif isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float)

    if normalize:
        images = images/255

    if not channels_first:
        if len(images.shape) == 4:
            images = images.permute(0,3,1,2)
        else:
            images = images.permute(2,0,1)

    grid = vutils.make_grid(images, nrow=nrow)
    vutils.save_image(grid, path)

def save_minigrid_levels_to_images(path, levels, labels=None, tile_size=32, n_per_row=8):

    images = minigrid_to_image(levels, tile_size=tile_size)
    if labels is None:
        save_images(images, path, channels_first=False, normalize=True, nrow=n_per_row)
    else:
        images = make_grid_with_labels(images, labels, nrow=n_per_row, normalize=True, limit=None,
                                       channels_first=False)
        vutils.save_image(images, path)


def minigrid_to_image(levels:List[np.ndarray], tile_size=32):
    if isinstance(levels, list):
        levels = np.array(levels)
    grid = Grid(levels.shape[-3], levels.shape[-2])
    images = []
    for level in levels:
        decoded_grid = grid.decode(level)[0]
        img = decoded_grid.render(tile_size=tile_size)
        images.append(img)
    return np.array(images)


def make_labels(dgps, is_dataset, scores, secondary_scores, grounded_values, seeds):
    labels = [f'dgp={dgps[i]} \n' \
              f'is_d={is_dataset[i]} \n' \
              f'sc={scores[i]:.3e} \n' \
              f'sc2={secondary_scores[i]:.3e} \n' \
              f'maxR={grounded_values[i]:.3e} \n' \
              f's={seeds[i]}' for i in range(len(dgps))]
    return labels


def make_grid_with_labels(tensor, labels, nrow=8, limit=20, padding=2,
                          normalize=False, v_range=None, scale_each=False, pad_value=0, channels_first=False):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W) [(B x H x W x C) if channels_first=False]
            or a list of images all of the same size.
        labels (list):  ( [labels_1,labels_2,labels_3,...labels_n]) where labels is Bx1 vector of some labels
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        v_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        channels_first (bool, optional): If ``False``, the input is expected in the (B x H x W x C) format.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if not isinstance(labels, list):
        raise ValueError
    else:
        pass
        #labels = np.asarray(labels).T[0]
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit, ::]

    import cv2
    font = 1
    fontScale = 2
    color = (255, 0, 0)
    thickness = 1

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        if isinstance(tensor[0], np.ndarray):
            # if list of numpy images, convert to a 4D mini-batch Tensor
            tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in tensor])
        else:
            tensor = torch.stack(tensor, dim=0)
    elif isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float)

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    if not channels_first:
        if len(tensor.shape) == 4:
            tensor = tensor.permute(0,3,1,2)
        else:
            tensor = tensor.permute(2,0,1)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if v_range is not None:
            assert isinstance(v_range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, v_range):
            if v_range is not None:
                norm_ip(t, v_range[0], v_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, v_range)
        else:
            norm_range(tensor, v_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    if nrow is None:
        xmaps = nmaps
    else:
        xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            working_tensor = tensor[k]
            if labels is not None:
                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.numpy(), (1, 2, 0)) * 255).astype('uint8'))
                y0, dy = int(tensor[k].shape[1] * 0.1), int(tensor[k].shape[1] * 0.1)
                for i, line in enumerate(labels[k].split('\n')):
                    y_t = y0 + i * dy
                    org = (0, y_t)
                    working_image = cv2.putText(working_image, line, org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = vtransforms.ToTensor()(working_image.get())
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid


def get_obs_at_index(obs, i):
    if isinstance(obs, dict):
        return {k: obs[k][i] for k in obs.keys()}
    else:
        return obs[i]


def set_obs_at_index(obs, obs_, i):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][i] = obs_[k].squeeze(0)
    else:
        obs[i] = obs_[0].squeeze(0)


def is_discrete_actions(env, adversary=False):
    if adversary:
        return env.adversary_action_space.__class__.__name__ == 'Discrete'
    else:
        return env.action_space.__class__.__name__ == 'Discrete'


def _make_env(args):
    env_kwargs = {'seed': args.seed}
    if args.singleton_env:
        env_kwargs.update({
            'fixed_environment': True})
    if args.env_name.startswith('CarRacing'):
        env_kwargs.update({
            'n_control_points': args.num_control_points,
            'min_rad_ratio': args.min_rad_ratio,
            'max_rad_ratio': args.max_rad_ratio,
            'use_categorical': args.use_categorical_adv,
            'use_sketch': args.use_sketch,
            'clip_reward': args.clip_reward,
            'sparse_rewards': args.sparse_rewards,
            'num_goal_bins': args.num_goal_bins,
        })

    if args.env_name.startswith('CarRacing'):
        # Hack: This TimeLimit sandwich allows truncated obs to be passed
        # up the hierarchy with all necessary preprocessing.
        env = gym_make(args.env_name, **env_kwargs)
        max_episode_steps = env._max_episode_steps
        reward_shaping = args.reward_shaping and not args.sparse_rewards
        assert max_episode_steps % args.num_action_repeat == 0
        return TimeLimit(CarRacingWrapper(env,
                grayscale=args.grayscale, 
                reward_shaping=reward_shaping,
                num_action_repeat=args.num_action_repeat,
                nstack=args.frame_stack,
                crop=args.crop_frame), 
            max_episode_steps=max_episode_steps//args.num_action_repeat)
    elif args.env_name.startswith('MultiGrid'):
        env = gym_make(args.env_name, **env_kwargs)
        if args.use_global_critic or args.use_global_policy:
            max_episode_steps = env._max_episode_steps
            env = TimeLimit(MultiGridFullyObsWrapper(env),
                max_episode_steps=max_episode_steps)
        return env
    else:
        return gym_make(args.env_name, **env_kwargs)


def create_parallel_env(args, adversary=True):
    is_multigrid = args.env_name.startswith('MultiGrid')
    is_car_racing = args.env_name.startswith('CarRacing')
    is_bipedalwalker = args.env_name.startswith('BipedalWalker')

    make_fn = lambda: _make_env(args)

    venv = ParallelAdversarialVecEnv([make_fn]*args.num_processes, adversary=adversary)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns)

    obs_key = None
    scale = None
    transpose_order = [2,0,1] # Channels first
    if is_multigrid:
        obs_key = 'image'
        scale = 10.0

    if is_car_racing:
        ued_venv = VecPreprocessImageWrapper(venv=venv) # move to tensor

    if is_bipedalwalker:
        transpose_order = None

    venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
            transpose_order=transpose_order, scale=scale)

    if is_multigrid or is_bipedalwalker:
        ued_venv = venv

    if args.singleton_env:
        seeds = [args.seed]*args.num_processes
    else:
        seeds = [i for i in range(args.num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv


def is_dense_reward_env(env_name):
    if env_name.startswith('CarRacing'):
        return True
    else:
        return False


def make_plr_args(args, obs_space, action_space):
    return dict( 
        seeds=[],
        obs_space=obs_space, 
        action_space=action_space, 
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
        strategy_support=args.level_replay_strategy_support,
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        replay_prob=args.level_replay_prob,
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature,
        staleness_support=args.staleness_support,
        secondary_strategy=args.level_replay_secondary_strategy,
        secondary_strategy_coef=args.level_replay_secondary_strategy_coef_start,
        secondary_score_transform=args.level_replay_secondary_score_transform,
        secondary_temperature=args.level_replay_secondary_temperature,
        secondary_strategy_support=args.level_replay_secondary_strategy_support,
        sample_full_distribution=args.train_full_distribution,
        seed_buffer_size=args.level_replay_seed_buffer_size,
        seed_buffer_priority=args.level_replay_seed_buffer_priority,
        random_new_seeds=args.ued_algo == 'seed_based_generation',
        use_dense_rewards=is_dense_reward_env(args.env_name),
        gamma=args.gamma
    )


def make_level_generator_args(args, device, level_info):
    assert args.generative_model_path is not None, "Must specify a generative model path"
    assert level_info, "Must specify level info"
    assert args.use_dataset, "DRED requires a dataset"
    assert args.use_editor, "Using a pre-trained generative model requires level editing enabled"
    num_interpolated_levels = args.num_level_pairs_for_interpolation * args.interpolations_per_pair
    assert args.interpolations_per_pair % 2 == 0, "--interpolations_per_pair must be even to determine a parent " \
                                                  "(i.e. closest neighbor in the latent space)"
    assert num_interpolated_levels >= args.num_processes, "Not enough interpolations created for the number of processes, " \
                                                          "increase --num_interpolated_pairs or --num_interpolations"
    return dict(
        device = device,
        level_info = level_info,
        model_checkpoint_path = args.generative_model_path,
        interpolations_per_pair = args.interpolations_per_pair,
        interpolation_scheme = args.interpolation_scheme,
        fixed_point_interpolation = args.fixed_point_interpolation,
        include_endpoints_in_interpolation = args.include_endpoints_in_interpolation,
        max_batch_size = args.generative_model_max_batch_size
    )


def get_node_features(graph:Union[dgl.DGLGraph, List[dgl.DGLGraph]], node_attributes:List[str]=None,
                      device=None, reshape:bool=True) -> Tuple[torch.Tensor, List[str]]:

    if device is None:
        if isinstance(graph, dgl.DGLGraph):
            device = graph.device
        else:
            device = graph[0].device

    # More efficient to rebatch graph
    if isinstance(graph, list) and isinstance(graph[0], dgl.DGLGraph):
        graph = dgl.batch(graph).to(device)

    if node_attributes is None:
        node_attributes = list(graph.ndata.keys())

    # Get node features
    Fx = []
    for attr in node_attributes:
        f = graph.ndata[attr]
        if reshape:
            f = f.reshape(graph.batch_size, -1)
        Fx.append(f)
    Fx = torch.stack(Fx, dim=-1).to(device)

    return Fx, node_attributes


def dgl_to_nx(graph: dgl.DGLGraph) -> nx.Graph:
    """Converts a DGLGraph to a NetworkX graph."""
    return nx.Graph(dgl.to_networkx(graph.cpu(), node_attrs=graph.ndata.keys()))


def nx_to_dgl(graph: nx.Graph) -> dgl.DGLGraph:
    """Converts a NetworkX graph to a DGLGraph."""
    if any([not isinstance(n, int) for n in graph.nodes()]):
        graph = nx.convert_node_labels_to_integers(graph)
    return dgl.from_networkx(graph, node_attrs=graph.nodes[0].keys())


def interpolate_between_pairs(pairs: List[Tuple[int, int]], data: torch.Tensor, num_interpolations: int,
                              scheme: str, endpoints=False) -> Tuple[torch.Tensor, List[Any]]:
    if endpoints:
        interpolation_points = np.linspace(0, 1, num_interpolations)
    else:
        interpolation_points = np.linspace(0, 1, num_interpolations + 2)[1:-1]
    interpolations = []
    for i, pair in enumerate(pairs):
        for loc in interpolation_points:
            if scheme == 'linear':
                interp = lerp(loc, data[pair[0]], data[pair[1]])
            elif scheme == 'polar':
                interp = slerp(loc, data[pair[0]], data[pair[1]])
            else:
                raise RuntimeError(f"Interpolation scheme {scheme} not recognised.")
            interpolations.append(interp)

    interpolations = torch.stack(interpolations).to(data)

    return interpolations


def lerp(val, low, high):
    return (1.0-val) * low + val * high


def slerp(val, low, high):
    dot = torch.dot(low/torch.linalg.norm(low), high/torch.linalg.norm(high))
    if torch.abs(dot) > 0.955:
        return lerp(val, low, high)
    omega = torch.arccos(torch.clip(torch.dot(low/torch.linalg.norm(low), high/torch.linalg.norm(high)), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return lerp(val, low, high) # L'Hopital's rule/LERP
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega) / so * high