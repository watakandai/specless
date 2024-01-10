from typing import List

from specless.specification.base import Specification


class MultiSpecifications(Specification):
    """Class for maintaining multiple specification models
    to make it work as a single specification model
    """

    def __init__(self, specifications: List[Specification]) -> None:
        super().__init__()
        self.specifications: List[Specification] = specifications

    def satisfy(self, demonstration) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()
