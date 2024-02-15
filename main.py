import csv
import logging
import os
import sys
import time

import gym

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)
from tpossp.solver.milp import MILPTSPTCSolver
from tpossp.tpo import TPO
from tpossp.tsconverter import TSConverter
from tpossp.tsptc import TSPTC
from wombats.automaton import active_automata
from wombats.systems.minigrid import ModifyActionsWrapper, StaticMinigridTSWrapper

EXPERIMENT_NAME = "minigrid"
GYM_MONITOR_LOG_DIR = os.path.join(HOME_DIR, f"experiment/{EXPERIMENT_NAME}/env_logs")


def runExperiment(n, num_constraint_ratio, epsilon):
    ENV_ID = "MiniGrid-TSPBenchmarkEnv-v0"
    env = gym.make(ENV_ID, num_locations=n, width=30, height=30, agent_start_pos=(1, 5))
    env = ModifyActionsWrapper(env, actions_type="simple_static")
    env = StaticMinigridTSWrapper(env, monitor_log_location=GYM_MONITOR_LOG_DIR)

    minigrid_TS = active_automata.get(
        automaton_type="TS", graph_data=env, graph_data_format="minigrid"
    )

    start_time = time.time()
    ts_converter = TSConverter(minigrid_TS, ignoring_obs_keys=["empty", "lava"])
    try:
        nodes, costs = ts_converter.to_tsp_nodes_and_costs()
    except Exception as e:
        print(e)
        print("Could not find a valid path")
        raise e
    nodesets = list(ts_converter.obs_to_nodes.values())

    service_times = {n: 0 for n in nodes}
    initial_nodes = [nodes[0]]
    global_constraints, local_constraints = TPO.generate_random_constraints(
        nodes,
        costs,
        service_times,
        initial_nodes,
        nodesets,
        num_constraint=num_constraint_ratio,
        epsilon=epsilon,
    )
    # global_constraints, local_constraints = {}, {}
    tpo = TPO(global_constraints, local_constraints)

    tsptc = TSPTC(nodes, costs, tpo)
    solver = MILPTSPTCSolver(tsptc)
    tours, cost = solver.solve()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("=" * 100)
    print(tours, cost, elapsed_time)
    print("=" * 100)
    return tours, cost, elapsed_time


if __name__ == "__main__":
    # Setup Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("logs.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    startNumLoc = 5
    every = 5
    maxLoc = 80 + every

    for iexperiment in range(10):
        csv_file_path = f"benchmark{iexperiment}.csv"
        header = [
            "#Locations",
            "#Agent",
            "#Constraint[%]",
            "epsilon",
            "Cost",
            "Time[s]",
        ]
        logger = logging.getLogger()
        logger.info(f"Header: {header}")

        with open(csv_file_path, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(header)

        for numLoc in reversed(range(startNumLoc, maxLoc, every)):
            for numAgent in [1, 5, 10, 20, 30, 40]:
                for epsilon in [10, 30, 50]:
                    for num_constraint_ratio in [0.25, 0.5, 0.75, 1.0]:
                        if numAgent > numLoc:
                            continue
                        try:
                            tours, cost, elapsed_time = runExperiment(
                                numLoc, num_constraint_ratio, epsilon
                            )
                            # result = [numLoc, numAgent, tours, cost, elapsed_time]
                            result = [
                                numLoc,
                                numAgent,
                                num_constraint_ratio,
                                epsilon,
                                cost,
                                elapsed_time,
                            ]
                        except Exception as e:
                            result = [
                                numLoc,
                                numAgent,
                                num_constraint_ratio,
                                epsilon,
                                -1,
                                -1,
                            ]
                            print(e)

                        with open(csv_file_path, "a") as f:
                            writer = csv.writer(f, delimiter=",")
                            writer.writerow(result)
