from .pdfa import PDFABuilder, PDFA  # NOQA
from .dfa import SafetyDFABuilder, SafetyDFA  # NOQA
from .fdfa import FDFABuilder, FDFA  # NOQA
from .base import Automaton  # NOQA
from .transition_system import (
    TSBuilder,  # NOQA
    TransitionSystem,  # NOQA
    MinigridTransitionSystem,  # NOQA
    build_transition_system,  # NOQA
)  # NOQA
from .product import ProductBuilder, Product  # NOQA
from .factory import AutomatonCollection  # NOQA

active_automata = AutomatonCollection()
active_automata.register_builder("DFA", SafetyDFABuilder())
active_automata.register_builder("PDFA", PDFABuilder())
active_automata.register_builder("FDFA", FDFABuilder())
active_automata.register_builder("TS", TSBuilder())
active_automata.register_builder("Product", ProductBuilder())
