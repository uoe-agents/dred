# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import logging

from collections import defaultdict, OrderedDict
from typing import List, Any, Dict, Union, Iterable, Tuple
import torch
import dgl
import networkx as nx
import numpy as np

import minigrid_level_generation.extra_util as util

logger = logging.getLogger(__name__)


def shortest_paths(graph: nx.Graph, source: int, target: int, num_paths: int = 1) -> List[Any]:
    """
    Compute shortest paths from source to target in graph.
    """
    assert num_paths >= 1

    # graph = nx.Graph(graph)

    if source == target or not nx.has_path(graph, source, target):
        return []

    if num_paths == 1:
        return [nx.shortest_path(graph, source, target, method="dijkstra")]
    else:
        return [p for p in nx.all_shortest_paths(graph, source, target, method='dijkstra')][:num_paths]


def shortest_path_length(graph: nx.Graph, source: int, target: int) -> int:
    """
    Compute shortest path length from source to target in graph.
    """
    return nx.shortest_path_length(graph, source, target)


def resistance_distance(graph: nx.Graph, source: int, target: int) -> int:
    """
    Compute resistance distance from source to target in graph. Graph must be strongly connected (1 single component)
    """
    if source == target or not nx.has_path(graph, source, target):
        return np.nan
    else:
        return nx.resistance_distance(graph, source, target)


def active_node_count(graph: nx.Graph) -> int:
    """
    Compute active node count in graph.
    """
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.number_of_nodes(graph)


def num_nav_nodes(graph: nx.Graph) -> int:
    """
    Compute navigable node count in graph. Input graph should have a grid
    Inputs:
    - graph: graph (nx.Graph)
    - start: starting node (int)
    Output: number of navigable nodes (int)
    """
    nav = len([graph.degree[i] for i in graph.nodes if graph.degree[i] != 0])
    return nav


def len_connected_component(graph: nx.Graph, source, target) -> int:
    if not source == target or nx.has_path(graph, source, target):
        return np.nan

    return len(nx.node_connected_component(graph, source))


def is_solvable(graph: nx.Graph, source, target) -> bool:
    """
    Check if the graph is solvable.
    """
    return nx.has_path(graph, source, target) and source != target


def object_count_vs_spl(graph: nx.Graph=None, target_node: Union[int, Tuple[int, int]]=None, object_type:'str'=None, navigable=True,
                        grid_size: Tuple[int, int] = (13, 13), depth: int = 3, spl=None) -> Dict[int, int]:
    """
    Compute object count vs SPL for a given graph and target node.
    Either provide as intput:
    - graph: graph (nx.Graph), target_node (int), object_type (str) -> computes spl and then the count of all nodes matching object type in graph
    - graph, graph(nx.Graph), spl: dict of shortest path lengths (dict), object_type (str) -> computes the count of all nodes matching object type in graph
    - spl: dict of shortest path lengths (dict) -> computes the count of all nodes provided in spl
    :param graph:
    :param target_node:
    :param object_type:
    :param navigable:
    :param grid_size:
    :param depth:
    :param spl:
    :return:
    """

    spl_object = defaultdict(int)
    if spl is None:
        spl_graph = dict(nx.single_target_shortest_path_length(graph, target_node))
    else:
        spl_graph = spl
    if not navigable:
        non_nav_nodes = [n for n in graph.nodes if graph.nodes[n]['wall'] == 1.0 or graph.nodes[n]['lava'] == 1.0]
        spl_graph = get_non_nav_spl(non_nav_nodes, spl_graph, grid_size, depth)
    for n, s in spl_graph.items():
        if object_type is None:
            spl_object[int(s)] += 1
        elif object_type is not None and graph.nodes[n][object_type] == 1.0:
            spl_object[int(s)] += 1
        else:
            pass
    return spl_object


def compile_spl_counts(spl_object_counts: List[Dict[int, int]], weights: np.ndarray = None) -> Dict[int, float]:
    """
    Compute SPL distribution from a list of SPL object counts.
    """
    assert weights.ndim == 1 if weights is not None else True
    weights = weights / weights.sum() if weights is not None else None
    assert len(spl_object_counts) == len(weights) if weights is not None else True
    spl_distribution = defaultdict(float)
    for m, spl_object_count in enumerate(spl_object_counts):
        if weights is not None:
            w = weights[m]
        else:
            w = 1.0
        for spl, count in spl_object_count.items():
            spl_distribution[spl] += w * count
    spl_domain = list(spl_distribution.keys())
    if len(spl_domain) == 0:
        return {}
    spl_max = int(max(spl_domain))
    for spl in range(spl_max + 1):
        if spl not in spl_distribution:
            spl_distribution[spl] = 0.0
        else:
            spl_distribution[spl] = spl_distribution[spl]
    spl_distribution = OrderedDict(sorted(spl_distribution.items()))
    return spl_distribution


def object_density(graph: nx.Graph, object_type: str, node_domain: List[str] = None) -> float:
    """
    Compute object density in graph.
    """
    if node_domain is None:
        total_nodes = len(graph.nodes)
    else:
        total_nodes = 0
    node_count = 0
    for n in graph.nodes:
        if graph.nodes[n][object_type] == 1.0:
            node_count += 1
        if node_domain is not None:
            for obj in node_domain:
                if graph.nodes[n][obj] == 1.0:
                    total_nodes += 1
                    break
    if total_nodes > 0:
        return node_count / total_nodes
    else:
        return 0.0


def compute_weighted_density(densities: np.ndarray, weigths: np.ndarray = None) -> float:
    """
    Compute weighted density from a list of densities.
    """
    if isinstance(densities, list):
        densities = np.array(densities)
    if isinstance(weigths, list):
        weigths = np.array(weigths)
    assert densities.ndim == 1
    assert weigths.ndim == 1 if weigths is not None else True
    assert len(densities) == len(weigths) if weigths is not None else True
    # ignore nan or inf values in densities
    if weigths is None:
        avg_density = float(np.mean(densities[np.isfinite(densities)]))
    else:
        w = weigths[np.isfinite(densities)] / np.sum(weigths[np.isfinite(densities)])
        avg_density = float(np.sum(w * densities[np.isfinite(densities)]))
    return avg_density


def prepare_graph(graph: Union[dgl.DGLGraph, nx.Graph], source: int=None, target: int=None)\
        -> Tuple[nx.Graph, bool, bool]:
    """Convert DGLGraph to nx.Graph and reduces it to a single component. Containing the source node.
    If the source node is not specified, the largest component is returned."""

    if isinstance(graph, dgl.DGLGraph):
        graph = dgl.to_networkx(graph.cpu(), node_attrs=graph.ndata.keys())
    elif isinstance(graph, nx.Graph):
        pass
    else:
        raise ValueError("graph must be a DGLGraph or nx.Graph")
    graph = nx.Graph(graph)
    inactive_nodes = [x for x, y in graph.nodes(data=True) if y['navigable'] < .5]
    graph.remove_nodes_from(inactive_nodes)
    nodes = set(graph.nodes)

    if source is not None:
        # Catch any unwanted exceptions here, but it should be handled if the graph supplied is valid
        if len(nodes) < 2 or source == target or source not in nodes or target not in nodes:
            valid = False
            connected = False
            return graph, valid, connected
        else:
            valid = True

        # if graph.degree[source] == 0 or graph.degree[target] == 0:
        #     valid = False
        # else:
        #     valid = True

        if nx.has_path(graph, source, target):
            connected = True
        else:
            connected = False

        component = nx.node_connected_component(graph, source)
    else:
        components = [graph.subgraph(c).copy() for c in sorted(nx.connected_components(graph), key=len, reverse=True) if
                      len(c) > 1]
        component = components[0]
        valid = True
        connected = True

    graph = graph.subgraph(component)
    return graph, valid, connected


def compute_metrics(graphs: Union[List[dgl.DGLGraph], List[nx.Graph]], labels=None) -> Dict[str, torch.Tensor]:

    if isinstance(graphs[0], dgl.DGLGraph):
        graphs = [util.dgl_to_nx(graph) for graph in graphs]

    metrics = {"valid": [], "solvable": [], "shortest_path": [], "resistance": [], "navigable_nodes":[]}
    for i, graph in enumerate(graphs):
        start_node = [n for n in graph.nodes if graph.nodes[n]['start'] == 1.0]
        goal_node = [n for n in graph.nodes if graph.nodes[n]['goal'] == 1.0]
        assert len(start_node) == 1
        assert len(goal_node) == 1
        start_node = start_node[0]
        goal_node = goal_node[0]
        if start_node != goal_node:
            metrics["valid"].append(True)
            solvable = nx.has_path(graph, start_node, goal_node)
        else:
            metrics["valid"].append(False)
            solvable = False
            # assert metrics["valid"][i] == False
        if not solvable:  # then these metrics do not make sense
            metrics["solvable"].append(False)
            metrics["shortest_path"].append(np.nan)
            metrics["resistance"].append(np.nan)
            metrics["navigable_nodes"].append(np.nan)
        else:
            subg = graph.subgraph(nx.node_connected_component(graph, start_node))
            metrics["solvable"].append(True)
            metrics["shortest_path"].append(shortest_path_length(subg, start_node, goal_node))
            metrics["resistance"].append(resistance_distance(subg, start_node, goal_node))
            metrics["navigable_nodes"].append(num_nav_nodes(subg))

    for metric in metrics:
        if metric in ["valid", "solvable"]:
            metrics[metric] = torch.tensor(metrics[metric], dtype=torch.bool)
        elif metric in ["shortest_path", "resistance", "navigable_nodes"]:
            metrics[metric] = torch.tensor(metrics[metric], dtype=torch.float)
        else:
            raise ValueError("Unknown metric")

    return metrics


def get_non_nav_spl(non_nav_nodes: List[Tuple[int, int]], spl_nav: Dict[Tuple[int, int], int],
                    grid_size: Tuple[int, int], depth: int = 3) -> Dict[Tuple[int, int], int]:
    neighbors_of_non_nav_nodes = get_neighbors(non_nav_nodes, list(spl_nav.keys()), grid_size)
    shortest_path_lengths = {}
    for node_ind, neighbors in enumerate(neighbors_of_non_nav_nodes):
        neighbors = [n for n in neighbors if n in spl_nav]
        if neighbors:
            pathlength = int(np.min([spl_nav[neighbor] for neighbor in neighbors])) + 1
        else:
            pathlength = None  # pathlength = np.max(list(shortest_path_lengths_nav.values())) + 1
        shortest_path_lengths[non_nav_nodes[node_ind]] = pathlength
    border_nodes = [n for n in shortest_path_lengths if shortest_path_lengths[n] is not None]
    nodes_to_remove = []
    for node in shortest_path_lengths:
        if shortest_path_lengths[node] is None:
            grid_graph = nx.grid_2d_graph(*grid_size)
            spl_ = dict(nx.single_target_shortest_path_length(grid_graph, node))
            if np.min([spl_[border_node] for border_node in border_nodes]) > depth:
                nodes_to_remove.append(node)
            else:
                spl_goal = [spl_[border_node] + shortest_path_lengths[border_node] for border_node in border_nodes]
                pathlength = np.min(spl_goal)
                shortest_path_lengths[node] = pathlength
    [shortest_path_lengths.pop(node) for node in nodes_to_remove]

    return shortest_path_lengths


def get_neighbors(nodes: List[Tuple[int, int]], neighbors_set: List[Tuple[int, int]],
                  grid_size: Tuple[int, int] = None):
    grid_graph = nx.grid_2d_graph(*grid_size)
    neighbors_grid = [list(grid_graph.neighbors(node)) for node in nodes]
    neighbors_grid = [list(set(neighbors_grid[i]) & set(neighbors_set)) for i in range(len(neighbors_grid))]
    return neighbors_grid
