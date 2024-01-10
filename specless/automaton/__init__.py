from .pdfa import PDFABuilder, PDFA
from .dfa import SafetyDFABuilder, SafetyDFA
from .fdfa import FDFABuilder, FDFA
from .base import Automaton
from .transition_system import (TSBuilder, TransitionSystem,
                                MinigridTransitionSystem)
from .product import ProductBuilder, Product
from .factory import AutomatonCollection

active_automata = AutomatonCollection()
active_automata.register_builder('DFA', SafetyDFABuilder())
active_automata.register_builder('PDFA', PDFABuilder())
active_automata.register_builder('FDFA', FDFABuilder())
active_automata.register_builder('TS', TSBuilder())
active_automata.register_builder('Product', ProductBuilder())
