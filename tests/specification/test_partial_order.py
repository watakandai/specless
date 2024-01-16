from specless.specification.base import Specification
from specless.specification.partial_order import PartialOrder


def test_construction():
    specification: Specification = PartialOrder()
    assert isinstance(specification, PartialOrder)


def test_satisfy():
    specification: Specification = PartialOrder()
    specification.add_edge(1, 2)
    specification.add_edge(2, 3)
    specification.add_edge(1, 4)
    specification.add_edge(3, 5)
    specification.add_edge(4, 5)

    assert specification.satisfy([1, 2, 3, 4, 5])
    assert specification.satisfy([1, 4, 2, 3, 5])
    assert not specification.satisfy([1, 3, 2, 4, 5])
