"""
=========
Automata
=========

This module provides functionality for working with automata.

The automaton module contains classes and functions for creating, manipulating, and simulating automata. It provides a high-level interface for defining automata and performing operations on them.

Classes:
- Automaton: Represents an automaton and provides methods for manipulating and simulating it.

Functions:
- create_automaton: Creates a new automaton with the specified properties.
- simulate_automaton: Simulates the given automaton on the provided input.

Usage:
To use this module, import it and create an instance of the Automaton class. You can then define the states, transitions, and initial/final states of the automaton using the provided methods. Once the automaton is defined, you can simulate it on different inputs using the simulate_automaton function.

Examples
--------
# Create an automaton
>>> automaton = Automaton()

# Define states
>>> automaton.add_state("q0")
>>> automaton.add_state("q1")
>>> automaton.add_state("q2")

# Define transitions
>>> automaton.add_transition("q0", "a", "q1")
>>> automaton.add_transition("q1", "b", "q2")
>>> automaton.add_transition("q2", "c", "q0")

# Set initial state
>>> automaton.set_initial_state("q0")

# Set final state
>>> automaton.set_final_state("q2")

# Simulate automaton on input
>>> input_string = "abc"
>>> result = simulate_automaton(automaton, input_string)
>>> print(result)  # Output: True

>>> input_string = "ab"
>>> result = simulate_automaton(automaton, input_string)
>>> print(result)  # Output: False
"""
