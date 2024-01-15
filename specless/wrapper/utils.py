import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.core import ObsType


def collect_demonstration(
    env: gym.Env,
    nsteps: int = 100,
    add_timestamp: bool = False,
    add_timestamp_func: Optional[Callable] = None,
) -> Tuple[List, bool, bool]:  # run simulation with random actions
    if add_timestamp and add_timestamp_func is None:

        def add_timestamp_func(s, t):
            if isinstance(s, list):
                return [t] + s
            elif isinstance(s, tuple):
                return (t, *s)
            else:
                msg = "Please pass a proper add_timestamp_func to add a timestamp to observations"
                raise Exception(msg)

    state: ObsType
    next_state: ObsType
    info: Dict
    reward: float
    terminated: bool
    truncated: bool
    (
        state,
        info,
    ) = env.reset()  # state = Dict{'x': gym.spaces.Space(), 'label': gym.spaces.Text()}
    t = 0
    if add_timestamp:
        demonstration = [add_timestamp_func(state, t)]
    else:
        demonstration = [state]
    for i in range(nsteps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        t += 1
        if add_timestamp:
            demonstration.append(add_timestamp_func(next_state, t))
        else:
            demonstration.append(next_state)
        state = next_state
        # Return only if succeeded/truncated
        if terminated or truncated:
            return demonstration, terminated, truncated
    env.close()
    return demonstration, False, False


def collect_demonstrations(
    env,
    num: int = 100,
    only_success: bool = False,
    only_failure: bool = False,
    only_finished: bool = False,
    only_unfinished: bool = False,
    timeout: int = 10,
    **kwargs,
) -> List[List]:
    msg = "Pick either only_success, only_failure, only_finished, or only_finished. `finished` includes success and failure."
    assert sum([only_success, only_failure, only_finished, only_unfinished]) < 2, msg

    if only_success:
        collect_types = ["success"]
    if only_failure:
        collect_types = ["failure"]
    if only_finished:
        collect_types = ["success", "failure"]
    if only_unfinished:
        collect_types = ["unfinished"]

    demonstrations: List = []
    start_time = time.time()

    while len(demonstrations) != num:
        # Collect a demonstration
        demo, terminated, truncated = collect_demonstration(env, **kwargs)

        # Decided whether to add the demonstration to the list
        if "success" in collect_types and terminated:
            demonstrations.append(demo)
        elif "failure" in collect_types and truncated:
            demonstrations.append(demo)
        else:
            demonstrations.append(demo)

        if time.time() - start_time > timeout:
            warnings.warn(
                f"Timeout. collect_demonstrations exceeded {timeout} seconds."
            )
            break

    return demonstrations
