"""
===================
Inference Algorithm
===================

Inference algorithms are the core of the Specless library. They are responsible for inferring a specification from demonstrations (dataset). The library provides a set of inference algorithms that can be used to infer different types of specifications. The inference algorithms are implemented as classes that inherit from the `InferenceAlgorithm` base class. Each inference algorithm implements the `infer` method, which takes a dataset as input and returns a specification.

The library provides the following inference algorithms:

1. Partial Order (PO) Inference Algorithm
2. Timed Partial Order (TPO) Inference Algorithm
3. Automata Inference Algorithm

The `POInferenceAlgorithm` infers a partial order specification from a list of traces. The `TPOInferenceAlgorithm` infers a timed partial order specification from a list of timed traces. The `AutomataInferenceAlgorithm` infers an automaton specification from a list of traces.

Here is an example of how to use the inference algorithms

```python
# Import the Specless library
import specless as sl

# Create a list of traces
traces = [["a", "b", "c"], ["a", "b", "b", "c"], ["a", "a", "b", "b", "c"]]

# Infer a partial order specification
inference = sl.POInferenceAlgorithm()
specification = inference.infer(traces)

# Infer a timed partial order specification
inference = sl.TPOInferenceAlgorithm()
specification = inference.infer(dataset)

# Infer an automaton specification
inference = sl.AutomataInferenceAlgorithm()
specification = inference.infer(dataset)

```

The `infer` method of the inference algorithms returns a specification object. The specification object can be used to generate a visualization of the inferred specification.

The inference algorithms can be used to infer specifications from different types of data, such as traces, timed traces, and datasets. The inferred specifications can be used to generate strategies for different types of systems, such as control systems, robotic systems, and game-playing systems.

The Specless library provides a flexible and extensible framework for inferring specifications from data. The library can be used to infer specifications for a wide range of applications, including system verification, system synthesis, and system analysis.
"""

from .base import InferenceAlgorithm  # NOQA
from .edsm import AutomataInferenceAlgorithm  # NOQA
from .partial_order import POInferenceAlgorithm  # NOQA
from .timed_partial_order import TPOInferenceAlgorithm  # NOQA
