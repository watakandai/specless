import os
from pathlib import Path

CURRENT_DIR = Path(os.getcwd())
EXAMPLE_DIR = CURRENT_DIR.parent
HOME_DIR = EXAMPLE_DIR.parent
os.chdir(HOME_DIR)

# additional imports
import queue
import random
import re
from collections import defaultdict
from functools import reduce

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

# from tpossp import utils  # TODO: COPY-PASTED from TPO. MUST DELETE OR MERGE INTO TPOSSP

CURRENT_DATA_DIR = os.path.join(CURRENT_DIR, "data")
CURRENT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
CURRENT_CONFIG_FILENAME = os.path.join(CURRENT_DIR, "config.json")


def graph_from_df(
    df,
    num_str: str = "No.",
    op_str: str = "Operation",
    act_str: str = "Activity",
    prec_str: str = "Precedence",
):
    # Select a color for an operation
    operations = df[op_str].unique()
    colors = list(mcolors.CSS4_COLORS.keys())

    # Select colros at equidistant
    indices = np.round(
        np.linspace(0, len(mcolors.CSS4_COLORS) - 1, len(operations))
    ).astype(int)
    selected_colors = {operations[i]: colors[idx] for i, idx in enumerate(indices)}

    g = nx.DiGraph()

    for _, row in df.iterrows():
        # Add Nodes
        curr = row[num_str]
        act = row[act_str]
        op = row[op_str]
        color = selected_colors[op]
        g.add_node(curr, Activity=act, Operation=op, fillcolor=color, style="filled")

        # Add Edges
        precs = row.get(prec_str)
        if precs is not None:
            values = re.findall(r"\d+", precs)
            if len(values) != 0:
                for value in values:
                    value = int(value)
                    g.add_edge(value, curr)

    return g


def listolist_to_list(list_of_list):
    return reduce(lambda a, b: a + b, list_of_list)


def merge_nodes(g, df, op_str: str = "Operation", num_str: str = "No."):
    # Select a color for an operation
    operations = df[op_str].unique()
    colors = list(mcolors.CSS4_COLORS.keys())

    # Select colros at equidistant
    indices = np.round(
        np.linspace(0, len(mcolors.CSS4_COLORS) - 1, len(operations))
    ).astype(int)
    selected_colors = {operations[i]: colors[idx] for i, idx in enumerate(indices)}

    G = nx.DiGraph()
    for operation in df[op_str].unique():
        nodes = df[df[op_str] == operation][num_str]
        operation_nodes[operation] = nodes
        innodes = listolist_to_list([list(g.predecessors(n)) for n in nodes])
        outnodes = listolist_to_list([list(g.successors(n)) for n in nodes])
        incomings[operation] = set([g.nodes[n][op_str] for n in innodes])
        outgoings[operation] = set([g.nodes[n][op_str] for n in outnodes])

        color = selected_colors[operation]
        G.add_node(operation, fillcolor=color, style="filled")

    for operation, sources in incomings.items():
        for source in sources:
            G.add_edge(source, operation)

    for operation, targets in outgoings.items():
        for target in targets:
            G.add_edge(operation, target)

    # remove self loop
    for operation in df[op_str].unique():
        if operation in G[operation]:
            G.remove_edge(operation, operation)

    trG = nx.transitive_reduction(G)
    for n in trG.nodes():
        nodes = list(G.nodes())
        for k, v in G.nodes[n].items():
            trG.nodes[n][k] = v
    return trG


def sample_trace(
    G, means: dict, stds: dict, max_thresholds: dict, set_bound: bool = False
):
    """
    Sample traces from a TPO.
    """

    inits = [n for n in G.nodes() if len(list(G.predecessors(n))) == 0]

    Q = queue.Queue()
    for init in inits:
        Q.put(init)
    visitedE = defaultdict(lambda: [])
    times = {}

    trace = []
    visited_events = []

    while not Q.empty():
        curr_node = Q.get()

        if any([p not in visitedE[curr_node] for p in G.predecessors(curr_node)]):
            continue

        curr_time = max([0.0] + [times[p] for p in visitedE[curr_node]])
        duration = np.random.normal(means[curr_node], stds[curr_node])
        duration = max(0, duration)
        if set_bound:
            duration = min(duration, max_thresholds[curr_node])
        curr_time += duration

        # Record Time
        times[curr_node] = curr_time
        if curr_node not in visited_events:
            trace.append((curr_time, curr_node))
            visited_events.append(curr_node)

        # Move onto the next node
        next_nodes = list(G.successors(curr_node))
        random.shuffle(next_nodes)

        for next_node in next_nodes:
            visitedE[next_node].append(curr_node)
            Q.put(next_node)

    return trace


def sample_traces(ntrace: int, G, means, stds, max_thresholds, set_bound):
    traces = []
    for i in range(ntrace):
        traces.append(sample_trace(G, means, stds, max_thresholds, set_bound))
    return traces


def load_aircraft_turnaround_data():
    filename = os.path.join(CURRENT_DATA_DIR, "ground_services_by_operations.csv")
    df = pd.read_csv(filename)
    filename = os.path.join(CURRENT_DATA_DIR, "duration.csv")
    duration_df = pd.read_csv(filename)

    # Add 6 to 7
    index = df[df["No."] == 7].index
    df.loc[index, "Precedence"] = df.loc[index, "Precedence"] + ",6"
    # Add 10 to 11
    index = df[df["No."] == 11].index
    df.loc[index, "Precedence"] = df.loc[index, "Precedence"] + ",10"
    # Add 16 to 17
    index = df[df["No."] == 17].index
    df.loc[index, "Precedence"] = df.loc[index, "Precedence"] + ",16"
    # Add 29 to 30
    index = df[df["No."] == 30].index
    df.loc[index, "Precedence"] = df.loc[index, "Precedence"] + ",29"
    # sink states (7, 15, 18, 20) to the last node 44.
    index = df[df["No."] == 44].index
    df.loc[index, "Precedence"] = df.loc[index, "Precedence"] + ",7,15,18,20,41"
    pd.set_option("display.max_rows", df.shape[0] + 1)

    operation_nodes = {}
    incomings = {}
    outgoings = {}

    g = graph_from_df(df)

    means = {}
    stds = {}
    max_thresholds = {}
    for i, row in duration_df.iterrows():
        if i == 0:
            continue
        operation = row.index[1]
        mean = row.index[2]
        std = row.index[3]
        delay = row.index[4]
        means[row[operation]] = int(row[mean])
        stds[row[operation]] = int(row[std])
        max_thresholds[row[operation]] = int(row[delay])
    print(means)
    print(stds)

    G = merge_nodes(g, df)
    traces = sample_traces(10000, G, means, stds, max_thresholds, set_bound=True)

    return traces


if __name__ == "__main__":
    load_aircraft_turnaround_data()
