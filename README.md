
<p align="center">
    <img src="https://github.com/watakandai/specless/assets/11141442/315fc857-aa58-46a3-89a9-bdbbf7c68fcf" width="200" height="200">
</p>


# specless (SPECification LEarning and Strategy Synthesis)

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
demonstrations = sl.TraceDataset([
    [(symbol1, symbol2, ..., symboln)],             # trace 1
    [(symbol1, symbol2, ..., symbolm)],             # trace 2
    [(symbol1, symbol2, ..., symboll)],             # trace 3
])
# or load from a file
demonstrations = sl.TraceDataset(pd.read_csv(csv_filename))

# Run the inference
inference = sl.TPOInference()
specification = inference.infer(demonstrations)     # returns a Specification

print(specification)                                # prints the specification
sl.save_graph(specification, filenpath='spec')      # exports the specification to a file
sl.draw_graph(specification, png_filepath='spec')   # drawws the specification to a file
```

#### Demonstrations can be obtained by simulating runs in an environment.
The environment is based on the OpenAI Gym library (or more specifically, [PettingZoo](https://pettingzoo.farama.org/index.html))
```python
import gym
env = gym.make('MiniGrid-TSPBenchmarkEnv-v0')       # define an openai gym env

def collect_demonstration(nsteps = 100):            # run simulation with random actions
    state = env.reset()                             # state = Dict{'x': gym.spaces.Space(), 'label': gym.spaces.Text()}
    demonstration = []
    for i in range(nsteps):
        action = env.action_space.random()
        next_state, reward, terminated, truncated, info = env.step(action)
        demonstration.append((state, action, next_state, reward, terminated, truncated, info))
        if terminated or truncated:
            observation, info = env.reset()
        state = next_state
    env.close()
    return demonstration

demonstrations = [collect_demonstration() for i in range(10)]
demonstrations = sl.MDPDataset(demonstrations)      # Collect Demonstrations
```

#### In case anyone wants to manually specify a specification, specify a LTL<sub>f</sub> formula:
Internally, we use the LTL<sub>f</sub>2DFA package AND/OR `spot` package.
```python
import specless as sl  # or laod from specless.specbuilder import LTLfBuilder
specbuilder = sl.LTLfBuilder(engine='ltlf2dfa')         # Choose an engine
specification = specbuilder.build("G(a -> X b)")        # Translate a LTLf formula to specification class
```


- Once the specification is obtained, synthesize a strategy:
```python
import gym
import specless as sl
# or from specless.specbuilder import LTLfBuilder
# or from specless.synthesis import TSPSynthesis

env = gym.make('MiniGrid-TSPBenchmarkEnv-v0')               # Define an env
specbuilder = sl.LTLfBuilder(engine='ltlf2dfa')             # Choose an engine
specification = specbuilder.build("G(a -> X b)")            # Translate a LTLf formula to specification class
synthesizer = sl.TSPSynthesis()                             # Set parameters at initialization
strategy = synthesizer.synthesize(specification, env)       # Run the synthesis Algorithm

print(strategy)
sl.save_graph(strategy, path='./strategy')
```

#### You can use the strategy in an env like
```python
state = env.reset()
while not (terminated or truncated):
    action = strategy.action(state)                         # Stategies in general should make decision based on an observed state
                                                            # PlanStrategy class is a feedforward strategy (precomputed plan)
                                                            # so it ignores whatever state it observes.
    env.step(action)
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
env.close()
```

### As a CLI Interface
With the [click](https://click.palletsprojects.com/en/8.1.x/) package, we exposed some functions as a commandline tool.

```python
demo2spec -f <path/to/file>
```

```python
synthesize -d <path/to/demo> OR -s <LTLf formula> AND -e <Gym env> AND -p <path/to/param>
```



## Development

If you want to contribute, set up your development environment as follows:

- Intall [Poetry](https://python-poetry.org)
- Clone the repository: `git clone https://github.com/watakandai/specless.git && cd specless`
- Install the dependencies: `poetry shell && poetry install`

## Tests

To run tests: `tox`

To run only the code tests: `tox`

## Docs

Locally, run `make html` inside the `docs` directory.

Once you are ready, make a pull request and the documentations are build automatically with GitHub Actions.
See `.github/generate-documentation.yml`.

## License

Apache 2.0 License

Copyright 2023- KandaiWatanabe

