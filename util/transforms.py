# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import itertools
import logging
import dgl
import einops
import networkx as nx
import numpy as np
import torchvision
from typing import Union, List, Dict, Tuple
from omegaconf import DictConfig
from collections import defaultdict, OrderedDict
from envs.multigrid.multigrid import Grid
import torch

from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX as Minigrid_OBJECT_TO_IDX, \
    IDX_TO_OBJECT as Minigrid_IDX_TO_OBJECT, COLOR_TO_IDX as Minigrid_COLOR_TO_IDX, Lava, Wall, Goal
from envs.multigrid.multigrid import Floor

import util

logger = logging.getLogger(__name__)

# This may lead to objects not being given the correct color if these 2 conditions are met:
# - Non default object colors are used within minigrid/multigrid.
# - We do not explicitly provide the new color_config when calling dense_graph_to_minigrid() / graph_features_to_minigrid()
# to convert a generated graph to a minigrid instance encoding.
MINIGRID_COLOR_CONFIG = {
    'empty': None,
    'agent': None,
    'wall' : Wall().color,
    'goal' : Goal().color,
    'lava' : Lava().color,
    'floor' : Floor().color,
    }

DENSE_GRAPH_NODE_ATTRIBUTES = ['navigable', 'non_navigable', 'start', 'goal', 'empty', 'wall', 'moss', 'lava']

DENSE_GRAPH_ATTRIBUTE_TO_OBJECT = {
    'empty': 'empty',
    'start': 'agent', # we encode the "start" attribute as an agent object as start is not a minigrid object
    'goal' : 'goal',
    'moss' : 'floor', # we encode the "moss" attribute as a floor object as moss is not a minigrid object
    'wall' : 'wall',
    'lava' : 'lava',
    'navigable': None,
    'non_navigable': None,
    }

OBJECT_TO_DENSE_GRAPH_ATTRIBUTE = {
    'empty': ('navigable', 'empty'),
    'start': ('navigable', 'start'),
    'agent': ('navigable', 'start'),
    'goal' : ('navigable', 'goal'),
    'floor' : ('navigable', 'moss'),
    'wall' : ('non_navigable', 'wall'),
    'lava' : ('non_navigable', 'lava'),
    }

class BinaryTransform(object):
    def __init__(self, thr):
        self.thr = thr

    def __call__(self, x):
        return (x >= self.thr).to(x)  # do not change the data type or device


class Nav2DTransforms:

    @staticmethod
    def minigrid_to_dense_graph(minigrids: Union[np.ndarray, List[MiniGridEnv]],
                                to_dgl=False,
                                make_batch=False,
                                node_attr=None,
                                edge_config=None) -> Union[List[dgl.DGLGraph], List[nx.Graph], dgl.DGLGraph]:
        if isinstance(minigrids[0], np.ndarray):
            minigrids = np.array(minigrids)
            layouts = minigrids[..., 0]
        elif isinstance(minigrids[0], MiniGridEnv):
            layouts = [minigrid.grid.encode()[..., 0] for minigrid in minigrids]
            for i in range(len(minigrids)):
                layouts[i][tuple(minigrids[i].agent_pos)] = Minigrid_OBJECT_TO_IDX['agent']
            layouts = np.array(layouts)
        else:
            raise TypeError(f"minigrids must be of type List[np.ndarray], List[MiniGridEnv], "
                            f"List[MultiGridEnv], not {type(minigrids[0])}")
        graphs, _ = Nav2DTransforms.minigrid_layout_to_dense_graph(layouts,
                                                                   to_dgl=to_dgl,
                                                                   make_batch=make_batch,
                                                                   remove_border=True,
                                                                   node_attr=node_attr,
                                                                   edge_config=edge_config)
        return graphs

    @staticmethod
    def minigrid_layout_to_dense_graph(layouts: np.ndarray,
                                       to_dgl=False,
                                       make_batch=False,
                                       remove_border=True,
                                       node_attr=None,
                                       edge_config=None,
                                       to_dgl_edge_g=False) -> \
            Tuple[Union[dgl.DGLGraph, List[dgl.DGLGraph], List[nx.Graph]], Dict[str, List[nx.Graph]]]:

        assert layouts.ndim == 3, f"Wrong dimensions for minigrid layout, expected 3 dimensions, got {layouts.ndim}."

        # Remove borders
        if remove_border:
            layouts = layouts[:, 1:-1, 1:-1]  # remove edges
        dim_grid = layouts.shape[1:]

        # Get the objects present in the layout
        objects_idx = np.unique(layouts)
        object_instances = [Minigrid_IDX_TO_OBJECT[obj] for obj in objects_idx]
        assert set(object_instances).issubset({"empty", "wall", "goal", "agent", "lava", "floor"}), \
            f"Unsupported object(s) in minigrid layout. Supported objects are: " \
            f"empty, wall, start, goal, agent, lava, floor. Got {object_instances}."

        # Get location of each object in the layout
        object_locations = {}
        for obj in object_instances:
            object_locations[obj] = defaultdict(list)
            ids = list(zip(*np.where(layouts == Minigrid_OBJECT_TO_IDX[obj])))
            for tup in ids:
                object_locations[obj][tup[0]].append(tup[1:])
            for m in range(layouts.shape[0]):
                if m not in object_locations[obj]:
                    object_locations[obj][m] = []
            object_locations[obj] = OrderedDict(sorted(object_locations[obj].items()))
        if 'start' not in object_instances and 'agent' in object_instances:
            object_locations['start'] = object_locations['agent']
        if 'agent' not in object_instances and 'start' in object_instances:
            object_locations['agent'] = object_locations['start']

        # Create one-hot graph feature tensor
        graph_feats = {}
        object_to_attr = OBJECT_TO_DENSE_GRAPH_ATTRIBUTE
        for obj in object_instances:
            for attr in object_to_attr[obj]:
                if attr not in graph_feats and attr in node_attr:
                    graph_feats[attr] = torch.zeros(layouts.shape)
                loc = list(object_locations[obj].values())
                assert len(loc) == layouts.shape[0]
                for m in range(layouts.shape[0]):
                    if loc[m]:
                        loc_m = torch.tensor(loc[m])
                        graph_feats[attr][m][loc_m[:, 0], loc_m[:, 1]] = 1
        for attr in node_attr:
            if attr not in graph_feats:
                graph_feats[attr] = torch.zeros(layouts.shape)
            graph_feats[attr] = graph_feats[attr].reshape(layouts.shape[0], -1)

        graphs, edge_graphs = Nav2DTransforms.features_to_dense_graph(graph_feats, dim_grid, edge_config, to_dgl,
                                                                       make_batch, to_dgl_edge_g)

        return graphs, edge_graphs

    @staticmethod
    def features_to_dense_graph(features: Dict[str, torch.Tensor],
                                dim_grid: tuple,
                                edge_config: DictConfig = None,
                                to_dgl=False,
                                make_batch=False,
                                to_dgl_edge_g=False) \
            -> Tuple[Union[dgl.DGLGraph, List[dgl.DGLGraph], List[nx.Graph]], Dict[str, List[nx.Graph]]]:

        graphs = []
        edge_graphs = defaultdict(list)
        for m in range(features[list(features.keys())[0]].shape[0]):
            g_temp = nx.grid_2d_graph(*dim_grid)
            g = nx.Graph()
            g.add_nodes_from(sorted(g_temp.nodes(data=True)))
            for attr in features:
                nx.set_node_attributes(g, {k: v for k, v in zip(g.nodes, features[attr][m].tolist())}, attr)
            if edge_config is not None:
                edge_layers = Nav2DTransforms.get_edge_layers(g, edge_config, list(features.keys()), dim_grid)
                for edge_n, edge_g in edge_layers.items():
                    g.add_edges_from(edge_g.edges(data=True), label=edge_n)
                    if to_dgl_edge_g:
                        edge_g = util.nx_to_dgl(edge_g)
                    edge_graphs[edge_n].append(edge_g)
            if to_dgl:
                g = nx.convert_node_labels_to_integers(g)
                g = dgl.from_networkx(g, node_attrs=features.keys()).to(features[list(features.keys())[0]].device)
            graphs.append(g)

        if to_dgl and make_batch:
            graphs = dgl.batch(graphs)

        return graphs, edge_graphs

    @staticmethod
    def graph_features_to_minigrid(graph_features: Dict[str,torch.Tensor], level_info,
                                   color_config=None, device=None) -> np.ndarray:

        features = graph_features.copy()
        node_attributes = list(features.keys())

        if device is None:
            device = features[node_attributes[0]].device

        if color_config is None:
            color_config = MINIGRID_COLOR_CONFIG

        shape_no_padding = (features[node_attributes[0]].shape[-2], level_info['shape'][0] - 2,
                            level_info['shape'][1] - 2, level_info['shape'][2])
        for attr in node_attributes:
            features[attr] = features[attr].reshape(*shape_no_padding[:-1])
        grids = torch.ones(shape_no_padding, dtype=torch.int, device=device) * Minigrid_OBJECT_TO_IDX['empty']

        minigrid_object_to_encoding_map = {}  # [object_id, color, state]
        for feature in node_attributes:
            obj_type = DENSE_GRAPH_ATTRIBUTE_TO_OBJECT[feature]
            if obj_type is not None and obj_type not in minigrid_object_to_encoding_map.keys():
                if obj_type == "empty":
                    minigrid_object_to_encoding_map[obj_type] = [Minigrid_OBJECT_TO_IDX["empty"], 0, 0]
                elif obj_type == "agent" or obj_type == "start":
                    minigrid_object_to_encoding_map[obj_type] = [Minigrid_OBJECT_TO_IDX["agent"], 0, 0]
                else:
                    color_str = color_config[obj_type]
                    minigrid_object_to_encoding_map[obj_type] = [Minigrid_OBJECT_TO_IDX[obj_type],
                                                                 Minigrid_COLOR_TO_IDX[color_str], 0]

        # if 'start' not in minigrid_object_to_encoding_map.keys() and 'agent' in minigrid_object_to_encoding_map.keys():
        #     minigrid_object_to_encoding_map['start'] = minigrid_object_to_encoding_map['agent']
        # if 'agent' not in minigrid_object_to_encoding_map.keys() and 'start' in minigrid_object_to_encoding_map.keys():
        #     minigrid_object_to_encoding_map['agent'] = minigrid_object_to_encoding_map['start']

        for i, attr in enumerate(node_attributes):
            try:
                obj = DENSE_GRAPH_ATTRIBUTE_TO_OBJECT[attr]
                if obj is None:
                    continue
                mapping = minigrid_object_to_encoding_map[obj]
                grids[features[attr] == 1] = torch.tensor(mapping, dtype=torch.int, device=device)
            except KeyError:
                pass

        padding = torch.tensor(minigrid_object_to_encoding_map['wall'], dtype=torch.int).to(device)
        padded_grid = einops.rearrange(grids, 'b h w c -> b c h w')
        padded_grid = torchvision.transforms.Pad(1, fill=-1, padding_mode='constant')(padded_grid)
        padded_grid = einops.rearrange(padded_grid, 'b c h w -> b h w c')
        padded_grid[torch.where(padded_grid[..., 0] == -1)] = torch.tensor(list(padding), dtype=torch.int).to(
            device)

        grids = padded_grid.cpu().numpy().astype(level_info['dtype'])

        return grids

    @staticmethod
    def dense_graph_to_minigrid(graphs: Union[dgl.DGLGraph, List[dgl.DGLGraph], List[nx.Graph]],
                                level_info, color_config=None, device=None) -> np.ndarray:

        features, node_attributes = util.get_node_features(graphs, node_attributes=None, reshape=True)
        num_zeros = features[features == 0.0].numel()
        num_ones = features[features == 1.0].numel()
        assert num_zeros + num_ones == features.numel(), "Graph features should be binary"
        features_dict = {}
        for i, key in enumerate(node_attributes):
            features_dict[key] = features[..., i].float()
        grids = Nav2DTransforms.graph_features_to_minigrid(features_dict,
                                                           level_info=level_info,
                                                           color_config=color_config,
                                                           device=device)

        return grids

    @staticmethod
    def get_edge_layers(graph:nx.Graph, edge_config:DictConfig, node_attr:List[str], dim_grid:Tuple[int, int]) \
            -> Dict[str, nx.Graph]:

        navigable_nodes = ['empty', 'start', 'goal', 'moss']
        non_navigable_nodes = ['wall', 'lava']
        assert all([isinstance(n, tuple) for n in graph.nodes])
        assert all([len(n) == 2 for n in graph.nodes])

        def partial_grid(graph, nodes, dim_grid):
            non_grid_nodes = [n for n in graph.nodes if n not in nodes]
            g_temp = nx.grid_2d_graph(*dim_grid)
            g_temp.remove_nodes_from(non_grid_nodes)
            g_temp.add_nodes_from(non_grid_nodes)
            g = nx.Graph()
            g.add_nodes_from(graph.nodes(data=True))
            g.add_edges_from(g_temp.edges)
            return g

        def pair_edges(graph, node_types):
            all_nodes = []
            for n_type in node_types:
                all_nodes.append([n for n, a in graph.nodes.items() if a[n_type] >= 1.0])
            edges = list(itertools.product(*all_nodes))
            edged_graph = nx.create_empty_copy(graph, with_data=True)
            edged_graph.add_edges_from(edges)
            return edged_graph

        edge_graphs = {}
        for edge_ in edge_config.keys():
            if edge_ == 'navigable' and 'navigable' not in node_attr:
                edge_config[edge_].between = navigable_nodes
            elif edge_ == 'non_navigable' and 'non_navigable' not in node_attr:
                edge_config[edge_].between = non_navigable_nodes
            if edge_config[edge_].structure is None:
                edge_graphs[edge_] = pair_edges(graph, edge_config[edge_].between)
            elif edge_config[edge_].structure == 'grid':
                nodes = []
                for n_type in edge_config[edge_].between:
                    nodes += [n for n, a in graph.nodes.items() if a[n_type] >= 1.0 and n not in nodes]
                edge_graphs[edge_] = partial_grid(graph, nodes, dim_grid)
            else:
                raise NotImplementedError(f"Edge structure {edge_config[edge_].structure} not supported.")

        return edge_graphs

    @staticmethod
    def dense_graph_to_minigrid_render(graphs, level_info, tile_size=32):
        grids = Nav2DTransforms.dense_graph_to_minigrid(graphs, level_info)
        return Nav2DTransforms.minigrid_to_minigrid_render(grids, tile_size=tile_size)

    @staticmethod
    def minigrid_to_minigrid_render(grids, tile_size=32):
        images = []
        GridObj = Grid(grids.shape[1], grids.shape[2])
        for i, grid in enumerate(grids):
            grid = grid.transpose(1, 0, 2)
            GridObj = GridObj.decode(grid)[0]
            img = GridObj.render(tile_size=tile_size)
            images.append(img)

        images = np.array(images)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).to(torch.float)
        return images

    @staticmethod
    def grid_graph_to_graph(grid_graphs: Union[dgl.DGLGraph, List[dgl.DGLGraph], List[nx.Graph]], to_dgl=False,
                            rebatch=False, device=None, node_attr=None) -> Union[List[dgl.DGLGraph], List[nx.Graph]]:
        # However should not be necessary to use for dgl graphs as they get encoded as graphs anyway.
        if device is None:
            device = "cpu"
        if node_attr is None:
            node_attr = DENSE_GRAPH_NODE_ATTRIBUTES

        if isinstance(grid_graphs, dgl.DGLGraph):
            device = grid_graphs.device
            grid_graphs = dgl.unbatch(grid_graphs)
            if to_dgl:
                rebatch = True

        if isinstance(grid_graphs[0], dgl.DGLGraph):
            to_dgl = True

        graphs = []
        for g in grid_graphs:
            if isinstance(g, dgl.DGLGraph):
                g = dgl.to_networkx(g.cpu(), node_attrs=g.ndata.keys())
            g = nx.convert_node_labels_to_integers(g)  # however this should be done automatically by dgl
            if to_dgl:
                g = dgl.from_networkx(g, node_attrs=node_attr)
            else:
                g = nx.Graph(g)
            graphs.append(g)

        if rebatch:
            graphs = dgl.batch(graphs).to(device)

        return graphs

    @staticmethod
    def graph_to_grid_graph(graphs: Union[dgl.DGLGraph, List[dgl.DGLGraph], List[nx.Graph]], level_info) -> \
            List[nx.Graph]:

        if isinstance(graphs, dgl.DGLGraph):
            graphs = dgl.unbatch(graphs)

        grid_graphs = []
        for g in graphs:
            if isinstance(g, dgl.DGLGraph):
                g = dgl.to_networkx(g.cpu(), node_attrs=g.ndata.keys())
                g = nx.Graph(g)
                for node in g.nodes:
                    for key in g.nodes[node].keys():
                        g.nodes[node][key] = g.nodes[node][key].item()
            assert g.number_of_nodes() == (level_info['shape'][0] - 2) * (level_info['shape'][1] - 2), \
                "Number of nodes does not match level info."
            g = nx.convert_node_labels_to_integers(g)
            g = nx.relabel_nodes(g, lambda x: (x // (level_info['shape'][1] - 2), x % (level_info['shape'][1] - 2)))
            grid_graphs.append(g)

        return grid_graphs
