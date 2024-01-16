import copy
from typing import Callable, Dict, List

import graphviz as gv
import networkx as nx
from IPython.display import Image, display
from networkx.drawing.nx_pydot import _check_colon_quotes, to_pydot
from pydot import Dot


def save_graph(graph: nx.MultiDiGraph, filepath: str) -> None:
    """Save a class that is derived from a networkx MultiDiGraph

    Args:
        graph (nx.MultiDiGraph): _description_
        filepath (str): _description_
    """
    pass


def node_label_function(
    n: str,
    data: Dict,
    selected_keys: List[str] = None,
    ignore_keys: List[str] = ["obj"],
) -> str:
    """Node Label Function"""
    strings = []
    if selected_keys is None:
        selected_keys = list(data.keys())
    for key in selected_keys:
        if key not in data:
            continue
        if ignore_keys is not None and key in ignore_keys:
            continue
        val = data[key]
        if isinstance(val, float):
            strings.append(f"{key}: {val:.1f}")
        elif isinstance(val, str) or hasattr(val, "__str__"):
            strings.append(f"{key}: {val}")
    return f"{n}: \n" + "\n".join(strings)


def edge_label_function(
    u: str,
    v: str,
    data: Dict,
    selected_keys: List[str] = None,
    ignore_keys: List[str] = ["obj"],
) -> str:
    strings = []
    if selected_keys is None:
        selected_keys = list(data.keys())
    for key in selected_keys:
        if key not in data:
            continue
        if ignore_keys is not None and key in ignore_keys:
            continue
        val = data[key]
        if isinstance(val, float):
            strings.append(f"{key}: {val:.1f}")
        elif isinstance(val, str) or hasattr(val, "__str__"):
            strings.append(f"{key}: {val}")
    return f"{u}->{v}: \n" + "\n".join(strings)


def add_labels(
    graph: nx.MultiDiGraph,
    node_label_func: Callable = node_label_function,
    edge_label_func: Callable = edge_label_function,
) -> None:
    """Add Labels to the original graph"""
    if node_label_func is not None:
        for n, nodedata in graph.nodes(data=True):
            label = node_label_func(n, nodedata)
            graph.nodes[n].update({"label": label})

    if edge_label_func is not None:
        if isinstance(graph, nx.MultiDiGraph):
            for u, v, key, edgedata in graph.edges(data=True, keys=True):
                label = edge_label_func(u, v, edgedata)
                graph[u][v][key]["label"] = label

        elif isinstance(graph, nx.DiGraph):
            for u, v, edgedata in graph.edges(data=True):
                label = edge_label_func(u, v, edgedata)
                graph[u][v]["label"] = label


def draw_graph(
    graph: nx.DiGraph,
    filepath: str,
    should_display: bool = True,
    img_format: str = "png",
    cleanup: bool = True,
) -> None:
    """Draw a class that is derived from a networkx DiGraph

    Args:
        graph (nx.DiGraph): _description_
        filepath (str): _description_
        should_display (bool):  Display
        img_format (str): Image Format
    """
    graph_ = copy.deepcopy(graph)
    for node in graph_.nodes():
        if "obj" in graph_.nodes[node]:
            del graph_.nodes[node]["obj"]
    if isinstance(graph_, nx.MultiDiGraph):
        for source, target, key in graph.edges(keys=True):
            if "obj" in graph_[source][target][key]:
                del graph_[source][target][key]["obj"]
    else:
        for source, target in graph.edges():
            if "obj" in graph_[source][target]:
                del graph_[source][target]["obj"]

    graph_ = _get_pydot_representation(graph_)

    if filepath:
        graph_ = gv.Source(graph_.to_string())
        path = graph_.render(format=img_format, filename=filepath, cleanup=cleanup)

        if should_display:
            display(Image(filename=path))
    else:
        dot_string = graph_.to_string()

        if should_display:
            display(gv.Source(dot_string))


def _get_pydot_representation(graph: nx.MultiDiGraph) -> Dot:
    """
    converts the networkx graph to pydot and sets graphviz graph attributes
    :returns:   The pydot Dot data structure representation.
    :rtype:     pydot.Dot
    """
    graph = _check_and_modify_colon_quotes(graph)
    graph = to_pydot(graph)
    graph.set_splines(True)
    graph.set_nodesep(0.5)
    graph.set_sep("+25,25")
    graph.set_ratio(1)

    return graph


def _check_and_modify_colon_quotes(graph: nx.MultiDiGraph):
    for n, nodedata in graph.nodes(data=True):
        for attr, value in nodedata.items():
            if isinstance(value, int):
                graph.nodes[n].update({attr: f"{value}"})
            if isinstance(value, float):
                graph.nodes[n].update({attr: f"{value:.2f}"})
            if isinstance(value, str) and _check_colon_quotes(value):
                graph.nodes[n].update({attr: f'"{value:}"'})

    if isinstance(graph, nx.MultiDiGraph):
        for u, v, key, edgedata in graph.edges(data=True, keys=True):
            for attr, value in edgedata.items():
                if isinstance(value, int):
                    graph[u][v][key][attr] = f"{value}"
                if isinstance(value, float):
                    graph[u][v][key][attr] = f"{value:.2f}"
                if isinstance(value, str) and _check_colon_quotes(value):
                    value = f'"{value}"'
                    graph[u][v][key][attr] = value

    elif isinstance(graph, nx.DiGraph):
        for u, v, edgedata in graph.edges(data=True):
            for attr, value in edgedata.items():
                if isinstance(value, int):
                    graph[u][v][attr] = f"{value}"
                if isinstance(value, float):
                    graph[u][v][attr] = f"{value:.2f}"
                if isinstance(value, str) and _check_colon_quotes(value):
                    value = f'"{value}"'
                    graph[u][v][attr] = value

    return graph


def save_strategy(strategy, filepath: str) -> None:
    """Draw a strategy class

    Args:
        strategy (): _description_
        filepath (str): _description_
    """
    pass
