
<p align="center">
    <img src="https://github.com/mattn/vim-gist/assets/11141442/22091c57-bef9-4b1c-9a40-df1b77d53613" width="200" height="200">
</p>

# [WIP] specless (SPECification LEarning and Strategy Synthesis)

[![Documentation Status](https://readthedocs.org/projects/specless/badge/?version=latest)](https://specless.readthedocs.io/en/latest/?badge=latest)
[![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/watakandai/5f5c84f28e80b29f2f9ce92300859446/raw/covbadge.json)](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/watakandai/5f5c84f28e80b29f2f9ce92300859446/raw/covbadge.json)
[![PyPI Latest Release](https://img.shields.io/pypi/v/specless)](https://pypi.org/project/specless/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/specless)](https://pypi.org/project/specless/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



- **Websites** https://watakandai.github.io/specless/ and https://specless.readthedocs.io/en/latest/
- **Source code**: https://github.com/watakandai/specless.git
- **PyPI**: https://pypi.org/project/specless/


# Installation

- from PyPI

```bash
pip install specless
```

- from source
```bash
pip install git@github.com:watakandai/specless.git
```

- or clone and install.
```bash
git clone https://github.com/watakandai/specless.git
cd specless
pip install .
```

# Quickstart

You can use the `specless` package in two ways: as a library, and as a CLI tool.



### To infer a specification from demonstrations,

#### Parse a demonstration file:
```python
import specless as sl  # or load from specless.inference import TPOInference
import pandas as pd

# Manually prepare a list of demonstrations
demonstrations = [
    ["e1", "e2", "e3", "e4", "e5"],             # trace 1
    ["e1", "e4", "e2", "e3", "e5"],             # trace 2
    ["e1", "e2", "e4", "e3", "e5"],             # trace 3
]
dataset = sl.ArrayDataset(demnstrations, columns=["symbol"])
# or load from a file
csv_filename = "./examples/readme/example.csv"
dataset = sl.BaseDataset(pd.read_csv(csv_filename))

# Run the inference
inference = sl.TPOInferenceAlgorithm()
specification = inference.infer(dataset)            # returns a Specification

# Note: Not yet supported
print(specification)                                # prints the specification
sl.save_graph(specification, filenpath='spec')       # exports the specification to a file
sl.draw_graph(specification, png_filepath='spec')    # drawws the specification to a file
```

#### Demonstrations can be obtained by simulating runs in an environment.
The environment is based on the OpenAI Gym library (or more specifically, [PettingZoo](https://pettingzoo.farama.org/index.html))
```python
import gymnasium as gym
import gym_minigrid # To load minigrid envs
from specless.gym.utils import collect_demonstration

#
env = gym.make("MiniGrid-Empty-5x5-v0")
# Collect Demonstrations
demonstrations = [collect_demonstration(env) for i in range(10)]
# Convert them to a Dataset Class
demonstrations = sl.ArrayDataset(demonstrations, columns=["s1", "s2", ...]) # state labels
```

- Once the specification is obtained, synthesize a strategy:
```python
import gymnasium as gym
import gym_minigrid # To load minigrid envs (e.g., MiniGrid-Empty-5x5-v0)
import specless as sl
# or from specless.specparser import LTLfParser
# or from specless.synthesis import TSPSynthesis

env = gym.make("MiniGrid-Empty-5x5-v0")
# TODO: LTLf must be installed. Need a little bit of work.
specparser = sl.LTLfParser(engine='ltlf2dfa')             # Choose an engine
specification = specparser.parse("G(a -> X b)")            # Translate a LTLf formula to specification class
synthesizer = sl.TSPSynthesisAlgorithm()                  # Set parameters at initialization
strategy = synthesizer.synthesize(specification, env)      # Run the synthesis Algorithm

print(strategy)
sl.save_graph(strategy, path='./strategy')
```

#### You can use the strategy in an env like
```python
state, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = strategy.action(state) # Stategies make a decision given an observed state

    (next_state,
     reward,
     terminated,
     truncated,
     info) = env.step(action)       # PlanStrategy class is ******a** feedforward strategy.
                                    # It precomputs a plan at each **step** and does not
                                    # depend on the observed state.
    state = next_state
env.close()
```

### [Not yet Supported] As a CLI Interface
With the [click](https://click.palletsprojects.com/en/8.1.x/) package, we exposed some functions as a command line tool.

```python
demo2spec -f <path/to/file>
```

```python
synthesize -d <path/to/demo> OR -s <LTLf formula> AND -e <Gym env> AND -p <path/to/param>
```



## Development

If you want to contribute, set up your development environment as follows:

- Install [Poetry](https://python-poetry.org)
- Clone the repository: `git clone https://github.com/watakandai/specless.git && cd specless`
- Install the dependencies: `poetry shell && poetry install`

## Tests

To run tests: `tox`

To run only the code tests: `tox`

## Docs

Locally, run `make html` inside the `docs` directory.

Once you are ready, make a pull request and the documentations are built automatically with GitHub Actions.
See `.github/generate-documentation.yml`.

## License

Apache 2.0 License

Copyright 2023- KandaiWatanabe



# WIP
- https://pypi.org/project/gradio-pannellum/
- CLI


# TODO:
1. Create a wrapper (LabelMinigridWrapper) that labels an observed state. It must:
    1. accept any labeling function given by the user
    2. skip/ignore some states that are unnecessary (e.g., empty labels). This can be specified by the user by providing a list of labels that must be ignored or by providing a function that checks for unnecessary labels.
2. Update the MiniGridTransitionSystemWrapper with LabelMiniGridWrapper so that the step function is updated to return observations with the augmented information (e.g. labels).
3. Update the TransitionSystemBuilder with the new MiniGridTransitionSystemWrapper so that the transition system only includes desired labels!!!
4. Collect a demonstration (trace) from the TransitionSystem
5. Collect a timed trace from the TransitionSystem
6. Update
