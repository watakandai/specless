
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


### As a Library

To infer a specification from demonstrations,

- Parse a demonstrations file:
```python
from specless.inference import TPOInference
import pandas as pd

# Manually prepare a list of demonstrations
demonstrations = [
    [],
    [],
    [],
]
# or load from a file
demonstrations = pd.read_csv(csv_filename)

# Run the inference
inference = TPOInference()
specification = inference.infer(demonstraitons)     # returns a Specification

print(specification)                                # prints the specification
specification.draw(filename)                        # exports the specification to a file
```

- Demonstrations can be obtained by simulating runs in an environment.
The environment is based on the OpenAI Gym library (or more specifically, [PettingZoo](https://pettingzoo.farama.org/index.html))
```python
import gym
env = gym.make('MiniGrid-TSPBenchmarkEnv-v0')       # define an openai gym env

def collect_demonstration(nsteps = 100):            # run simulation with random actions
    state = env.reset()
    demonstration = []
    for i in range(nsteps):
        action = env.action_space.random()
        next_state, reward, terminated, truncated, info = env.step(action)
        demonstration.append(state, action, next_state, reward, terminated, truncated, info)
        if terminated or truncated:
            observation, info = env.reset()
        state = next_state
    env.close()
    return demonstration

demonstrations = [collect_demonstration() for i in range(10)]   # Collect Demonstrations
```

- In case anyone wants to manually specify a specification, specify a LTL<sub>f</sub> formula:
Internally, we use the LTL<sub>f</sub>2DFA package AND/OR `spot` package.
```python
from specless.specbuilder import LTLfBuilder
specification = LTLfBuilder.from_formula("G(a -> X b)")     # Translate a LTLf formula to specification class
```


- Once the specification is obtained, synthesize a strategy:
```python
import gym
from specless.specbuilder import LTLfBuilder
from specless.synthesis import TSPSynthesis
env = gym.make('MiniGrid-TSPBenchmarkEnv-v0')               # Define an env
specification = LTLfBuilder.from_formula("G(a -> X b)")     # Define a specification
synthesizer = TSPSynthesis()                                # Set parameters at initialization
strategy = synthesizer.syntheize(env, specification)        # Run the synthesis Algorithm

print(strategy)
strategy.draw()
```

You can use the strategy in an env like
```python
state = env.reset()
while not (terminated or truncated):
    action = strategy.action(state)
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
synthesize -d <path/to/demo> OR -s <LTLf formula> AND -e <Gym env> -p <path/to/param>
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

