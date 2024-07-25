
<p align="center">
    <img src="https://github.com/watakandai/specless/assets/11141442/ae4d1291-f959-4b46-b9f0-0fe55287467d" width="200" height="200">
</p>


#  specless (SPECification LEarning and Strategy Synthesis)

[![Deploy Documentation](https://github.com/watakandai/specless/actions/workflows/main.yml/badge.svg)](https://github.com/watakandai/specless/actions/workflows/main.yml)
[![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/watakandai/5f5c84f28e80b29f2f9ce92300859446/raw/covbadge.json)](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/watakandai/5f5c84f28e80b29f2f9ce92300859446/raw/covbadge.json)
[![PyPI Latest Release](https://img.shields.io/pypi/v/specless)](https://pypi.org/project/specless/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/specless)](https://pypi.org/project/specless/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



- **Websites** https://watakandai.github.io/specless/
- **Source code**: https://github.com/watakandai/specless.git
- **PyPI**: https://pypi.org/project/specless/


# Installation

<!-- - from PyPI

```bash
pip install specless
```
- from source
```bash
pip install git@github.com:watakandai/specless.git
``` -->

### Install poetry (Linux / Mac)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install specless & dependencies

```bash
# Clone locally
git clone https://github.com/watakandai/specless.git
cd specless

# Optional: Change Python version using pyenv
# pyenv install 3.9
# pyenv local 3.9

# Activate the virtual environment
python -m venv env
source env/bin/activate
# Install specless & dependencies
poetry install
```

Done!

# Quickstart

You can use the `specless` package in two ways: as a library, and as a CLI tool.



### To infer a specification from demonstrations,

#### Parse a demonstration file:
```python
>>> import specless as sl  # or load from specless.inference import TPOInference
>>> import pandas as pd

# Manually prepare a list of demonstrations
>>> demonstrations = [
...    ["e1", "e2", "e3", "e4", "e5"],             # trace 1
...    ["e1", "e4", "e2", "e3", "e5"],             # trace 2
...    ["e1", "e2", "e4", "e3", "e5"],             # trace 3
... ]

# Run the inference
>>> inference = sl.POInferenceAlgorithm()
>>> specification = inference.infer(demonstrations)  # returns a Specification

# prints the specification
>>> print(specification) # doctest: +ELLIPSIS
PartialOrder...

# exports the specification to a file
>>> sl.save_graph(specification, filepath='spec')

# drawws the specification to a file
>>> sl.draw_graph(specification, filepath='spec')
<IPython.core.display.Image object>

```

#### Demonstrations can be obtained by simulating runs in an environment.
The environment is based on the OpenAI Gym library (or more specifically, [PettingZoo](https://pettingzoo.farama.org/index.html))
```python
>>> import gymnasium as gym
>>> import gym_minigrid # To load minigrid envs
>>> import specless as sl

# Instantiate an env
>>> env = gym.make("MiniGrid-Empty-5x5-v0")
>>> env = sl.LabelMiniGridWrapper(env, labelkey="label")
>>> env = sl.SelectStateDataWrapper(env, columns=["label"])

>>> # Collect Demonstrations
>>> demonstrations = sl.collect_demonstrations(
...     env,
...     only_success=False,
...     add_timestamp=True,
...     num=10,
...     timeout=1000,
... )
```

- Once the specification is obtained, synthesize a strategy:
```python
>>> import gymnasium as gym
>>> import gym_minigrid # To load minigrid envs (e.g., MiniGrid-Empty-5x5-v0)
>>> import specless as sl

>> env = gym.make("MiniGrid-Empty-5x5-v0")

# Choose an engine
>> specparser = sl.LTLfParser(engine='ltlf2dfa')

# Translate a LTLf formula to specification class
>> specification = specparser.parse("G(a -> X b)")

# Set parameters at initialization
>> synthesizer = sl.TSPSynthesisAlgorithm()

# Run the synthesis Algorithm
>> strategy = synthesizer.synthesize(specification, env)

>> print(strategy)
>> sl.save_graph(strategy, filepath='./strategy')
```

#### You can use the strategy in an env like
```python
>> state, info = env.reset()
>> terminated, truncated = False, False
>> while not (terminated or truncated):
..    action = strategy.action(state) # Stategies make a decision given an observed state
..    (next_state,
..     reward,
..     terminated,
..     truncated,
..     info) = env.step(action)       # PlanStrategy class is ******a** feedforward strategy.
..                                    # It precomputs a plan at each **step** and does not
..                                    # depend on the observed state.
..    state = next_state
>> env.close()

```

### [Not yet Supported] As a CLI Interface
With the [click](https://click.palletsprojects.com/en/8.1.x/) package, we exposed some functions as a command line tool.

```python
demo2spec -f <path/to/file> -a tpo
```

```python
synthesize -d <path/to/demo> OR -s <LTLf formula> AND -e <Gym env> AND -p <path/to/param>
```


## Docker + VSCode
Use Dev Container.


## Development

If you want to contribute, set up your development environment as follows:

- Install [Poetry](https://python-poetry.org)
- Clone the repository: `git clone https://github.com/watakandai/specless.git && cd specless`
- Install the dependencies: `poetry shell && poetry install`

## Tests

To run all tests: `tox`

To run only the code tests: `tox -e py38` (or py39, py310, py311)

To run doctests, `tox -e doctest`

To obtain test coverages : `tox -e report`

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
1. Load demonstration from a file
2. Choose an inference algorithm
3. MUST SUPPORT AUTOMATA INFERENCE
4. Implement a CLI
