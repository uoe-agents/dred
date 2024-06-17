# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import logging

import os.path
import pickle
from typing import Any, Optional, Tuple, Dict, List, Union

import networkx as nx
import numpy as np
import torch
import dgl

from torch.utils.data import Dataset, SubsetRandomSampler, RandomSampler

from util import DotDict, Nav2DTransforms, dgl_to_nx
from envs.multigrid.multigrid import Grid

logger = logging.getLogger(__name__)


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


class EnvDataset(Dataset):
    """Environment Parameter Dataset.

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
    """

    base_folder = ""
    meta = {
        "filename": "dataset.meta"
    }

    # @memory_profiler.profile
    def __init__(
            self,
            root: str,
            train: bool = True,
            no_images=False,
            held_out_tasks: Optional[List[str]] = None,
    ) -> None:

        super(EnvDataset, self).__init__()

        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.no_images = no_images

        self._load_meta()
        dataset_path = os.path.join(self.root, self.base_folder)
        files = os.listdir(dataset_path)
        self.train_list = [f for f in files if f.endswith('.data') and not f.startswith('test')]
        self.test_list = [f for f in files if f.endswith('.data') and f.startswith('test')]

        self.train = train  # training set or test set

        if self.train:
            pickled_data = self.train_list
        else:
            pickled_data = self.test_list

        self.data: Any = []
        self.targets = []
        self.target_contents: Dict = {}
        self.batches_metadata: Any = []
        self.held_out_tasks = held_out_tasks

        if self.held_out_tasks is not None:
            logger.info("Held out tasks: {}".format(self.held_out_tasks))

        # now load the picked numpy arrays
        for file_name in pickled_data:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            self._load_data(file_path)

        if not self.data:
            raise FileNotFoundError("Dataset not found at specified location.")

        self.target2idx = {target.item(): idx for idx, target in enumerate(self.targets)}

    # @memory_profiler.profile
    def _load_data(self, file_path: str) -> None:

        try:
            meta_file_path = file_path + '.meta'
            extra_data_file_path = file_path + '.dgl.extra'
            with open(meta_file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                if isinstance(entry['label_contents'], DotDict):
                    entry['label_contents'] = entry['label_contents'].to_dict()
                if isinstance(entry['batch_meta'], DotDict):
                    entry['batch_meta'] = entry['batch_meta'].to_dict()

                ts = entry['label_contents']['task_structure'][0]
                assert all(ts == i for i in entry['label_contents']['task_structure']), \
                    f"Not all tasks are the same in batch {file_path}."
                if self.held_out_tasks is not None and ts in self.held_out_tasks:
                    return
                if self.data_type == 'graph':
                    if not os.path.exists(file_path): raise FileNotFoundError()
                    graphs, labels = dgl.load_graphs(file_path)
                    if os.path.exists(extra_data_file_path):
                        extra_data = dgl.load_graphs(extra_data_file_path)
                        extra_data = assemble_extra_data(extra_data)
                        entry['label_contents']['edge_graphs'] = extra_data['edge_graphs']

                    self.data.extend(graphs)
                    self.targets.extend(labels['labels'])
                if self.no_images:
                    try:
                        del entry["label_contents"]["images"]
                    except KeyError as e:
                        pass
                self.batches_metadata.append(entry["batch_meta"])
                try:
                    for key in entry["label_contents"].keys():
                        if key == '__dict__':  # handles DotDict objects
                            continue
                        if isinstance(entry["label_contents"][key], list):
                            if key not in self.target_contents:
                                self.target_contents[key] = entry["label_contents"][key]
                            else:
                                self.target_contents[key].extend(entry["label_contents"][key])
                        elif isinstance(entry["label_contents"][key], torch.Tensor):
                            if key not in self.target_contents:
                                self.target_contents[key] = entry["label_contents"][key]
                            else:
                                self.target_contents[key] = \
                                    torch.cat((self.target_contents[key], entry["label_contents"][key]))
                        elif isinstance(entry["label_contents"][key], np.ndarray):
                            val = torch.from_numpy(entry["label_contents"][key])
                            if key not in self.target_contents:
                                self.target_contents[key] = val
                            else:
                                self.target_contents[key] = torch.cat((self.target_contents[key], val))
                        elif isinstance(entry["label_contents"][key], dict):
                            if key not in self.target_contents:
                                self.target_contents[key] = entry["label_contents"][key]
                            else:
                                for k in entry["label_contents"][key].keys():
                                    if k not in self.target_contents[key]:
                                        self.target_contents[key][k] = entry["label_contents"][key][k]
                                    else:
                                        self.target_contents[key][k].extend(entry["label_contents"][key][k])
                        else:
                            raise ValueError("Unsupported type for target_contents")
                except KeyError as e:
                    raise KeyError(
                        f"{e} not found in {self.target_contents.keys()}. Mismatch in label contents "
                        f"across batch files")
        except FileNotFoundError as e:
            pass

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            self.dataset_metadata = pickle.load(infile, encoding="latin1")
            self.label_descriptors = self.dataset_metadata.config.label_descriptors
            self.data_type = self.dataset_metadata.config.data_type
        self.label_to_idx = {_class: i for i, _class in enumerate(self.label_descriptors)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is index of the target class.
        """
        sample, target = self.data[index], self.targets[index]

        return sample, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class MinigridDataset(EnvDataset):
    """Minigrid environment parameters dataset."""

    def __init__(
            self,
            root: str,
            train: bool = True,
            no_images=False,
            held_out_tasks: Optional[List[str]] = None,
    ) -> None:

        super().__init__(root, train, no_images, held_out_tasks)

    def get_task_structure_from_target(self, target) -> str:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['task_structure'][self.target2idx[target]]

    def get_resistance_distance_from_target(self, target, normalised=False) -> float:
        if isinstance(target, torch.Tensor):
            target = target.item()
        if not normalised:
            if self.dataset_metadata.metrics_normalised:
                norm_factor = self.dataset_metadata.config.label_descriptors_config.resistance['normalisation_factor']
            else:
                norm_factor = 1.0
            metric = self.target_contents['resistance'][self.target2idx[target]].item() * norm_factor
        else:
            if not self.dataset_metadata.metrics_normalised:
                logger.warning(
                    "Normalised metric requested, but metric is not normalised. Returning unnormalised metric.")
            metric = self.target_contents['resistance'][self.target2idx[target]].item()
        return metric

    def get_shortest_path_from_target(self, target, normalised=False) -> float:
        if isinstance(target, torch.Tensor):
            target = target.item()
        if not normalised:
            if self.dataset_metadata.metrics_normalised:
                norm_factor = self.dataset_metadata.config.label_descriptors_config.resistance['normalisation_factor']
            else:
                norm_factor = 1.0
            metric = int(np.round(self.target_contents['shortest_path'][self.target2idx[target]].item() * norm_factor))
        else:
            if not self.dataset_metadata.metrics_normalised:
                logger.warning(
                    "Normalised metric requested, but metric is not normalised. Returning unnormalised metric.")
            metric = self.target_contents['shortest_path'][self.target2idx[target]].item()
        return metric

    def get_navigable_nodes_from_target(self, target, normalised=False) -> float:
        if isinstance(target, torch.Tensor):
            target = target.item()
        if not normalised:
            if self.dataset_metadata.metrics_normalised:
                norm_factor = self.dataset_metadata.config.label_descriptors_config.resistance['normalisation_factor']
            else:
                norm_factor = 1.0
            metric = int(
                np.round(self.target_contents['navigable_nodes'][self.target2idx[target]].item() * norm_factor))
        else:
            if not self.dataset_metadata.metrics_normalised:
                logger.warning(
                    "Normalised metric requested, but metric is not normalised. Returning unnormalised metric.")
            metric = self.target_contents['navigable_nodes'][self.target2idx[target]].item()
        return metric

    def get_spl_from_target(self, target) -> Dict[Tuple[int, int], int]:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['shortest_path_dist'][self.target2idx[target]]

    def get_nav_graph_from_target(self, target, to_nx=True, to_gridgraph=False) -> Union[nx.Graph, dgl.DGLGraph]:
        if isinstance(target, torch.Tensor):
            target = target.item()
        graph = self.target_contents['edge_graphs']['navigable'][self.target2idx[target]]
        if to_nx:
            graph = dgl_to_nx(graph)
        if to_gridgraph:
            assert to_nx, "Cannot convert to gridgraph without converting to nx first."
            graph = \
            Nav2DTransforms.graph_to_grid_graph([graph], level_info=self.dataset_metadata.level_info)[0]

        return graph

    def get_level_encoding_from_target(self, target) -> np.ndarray:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['minigrid'][self.target2idx[target]].cpu().numpy()

    def get_moss_density_from_target(self, target) -> float:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['moss_density'][self.target2idx[target]]

    def get_lava_density_from_target(self, target) -> float:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['lava_density'][self.target2idx[target]]

    def get_spl_vs_moss_count_from_target(self, target) -> Dict[Tuple[int, int], int]:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['spl_vs_moss_count'][self.target2idx[target]]

    def get_spl_vs_lava_count_from_target(self, target) -> Dict[Tuple[int, int], int]:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['spl_vs_lava_count'][self.target2idx[target]]

    def get_spl_vs_wall_count_from_target(self, target) -> Dict[Tuple[int, int], int]:
        if isinstance(target, torch.Tensor):
            target = target.item()
        return self.target_contents['spl_vs_wall_count'][self.target2idx[target]]

    def pickle(self, path, label_id, format='level_embedding', level_info=None):

        if format == 'level_embedding':
            pass
        else:
            raise NotImplementedError

        if level_info is None:
            level_info = self.dataset_metadata['level_info']
        task_structure = self.target_contents['task_structure'][label_id]
        seed = self.target_contents['seed'][label_id]

        graph = self.data[label_id]
        encoding = Nav2DTransforms.dense_graph_to_minigrid([graph], level_info=level_info)[0]
        data = {
            'task_structure': task_structure,
            'encoding': encoding,
            'seed': seed,
            'level_info': level_info
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)


class MinigridDatasetSampler():

    def __init__(self, dataset_path, reload_level_encodings=False):
        dataset_path = os.path.expandvars(os.path.expanduser(dataset_path))
        self.dataset_train = MinigridDataset(dataset_path, train=True, no_images=True)
        self.dataset_test = MinigridDataset(dataset_path, train=False, no_images=True)

        if reload_level_encodings:
            for dataset in [self.dataset_train, self.dataset_test]:
                levels = dataset.target_contents['minigrid'].cpu().numpy()
                new_lvls = []
                for lvl in levels:
                    grid = Grid(lvl.shape[0], lvl.shape[1])
                    grid.set_encoding(lvl)
                    new_lvl = grid.encode()
                    new_lvls.append(new_lvl)
                dataset.target_contents['minigrid'] = torch.tensor(np.stack(new_lvls))

    def sample_train_levels(self, num_levels):
        return self._sample_levels(num_levels, 'train')

    def sample_test_levels(self, num_levels):
        return self._sample_levels(num_levels, 'test')

    def _sample_levels(self, num_samples, split):
        if split == 'train':
            dataset = self.dataset_train
        elif split == 'test':
            dataset = self.dataset_test
        else:
            raise ValueError(f"Invalid split {split}")

        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        labels = [dataset[idx][1].tolist() for idx in indices]
        levels = np.stack([dataset.get_level_encoding_from_target(lbl) for lbl in labels])

        return levels, labels

    @property
    def train_graphs(self):
        return self.dataset_train.data

    @property
    def train_levels(self):
        return self.dataset_train.target_contents['minigrid'].cpu().numpy()

    @property
    def train_labels(self):
        return [lbl.item() for lbl in self.dataset_train.targets]

    @property
    def test_levels(self):
        return self.dataset_test.target_contents['minigrid'].cpu().numpy()

    @property
    def test_labels(self):
        return [lbl.item() for lbl in self.dataset_test.targets]
