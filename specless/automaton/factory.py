# local packages
from specless.factory.object_factory import ObjectFactory

from .base import Automaton


class AutomatonCollection(ObjectFactory):
    """
    registering the builders for the different types of automaton objects
    with a more readable interface to our generic factory class.
    """

    def get(self, automaton_type: str, **config_data) -> Automaton:
        """
        return an instance of an automaton given the automaton_type and the
        config_data.

        If the automaton has already been intialized with the same
        configuration data, it will return the already-initialized instance of
        it

        :param      automaton_type:  The automaton type
        :param      config_data:     The keywords arguments to pass to the
                                     specific automaton builder class

        :returns:   an intialized / active reference to the desired type of
                    automaton object
        """

        return self.create(automaton_type, **config_data)
