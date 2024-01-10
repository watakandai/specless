# local packages
from .builder import Builder


class ObjectFactory:
    """
    Generic object factory leveraging the generic Builder interface
    see: https://realpython.com/factory-method-python/
    """

    def __init__(self) -> 'ObjectFactory':
        """
        Constructs a new instance of the ObjectFactory
        """

        self._builders = {}

    def register_builder(self, key: str, builder: Builder) -> None:
        """
        adds the builder object to the internal builder dictionary

        effects: the _builders dictionary has the key-builder pair appended

        :param      key:      The _builders dict key reffering to the builder
        :type       key:      string
        :param      builder:  The Builder object
                              This can be any function,
                              class or object implementing the .__call__()
                              method
        :type       builder:  Builder
        """

        self._builders[key] = builder

    def create(self, key: str, **kwargs):
        """
        Returns an instance object built with the keyed builder key and the
        constructor arguments in kwargs

        :param      key:         The _builders dict key reffering to the
                                 builder
        :type       key:         string
        :param      kwargs:      The keywords arguments needed by the builder
                                 specified by key
        :type       kwargs:      dictionary

        :returns:   A concrete object built by the builder specified with key
        :rtype:     who knows lol

        :raises     ValueError:  given key must match an existing builder in
                                 _builders
        """

        builder = self._builders.get(key)

        if not builder:
            raise ValueError(key)

        return builder(**kwargs)
