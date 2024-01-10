from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder


def test_construction():
    specification: Specification = TimedPartialOrder()
    assert isinstance(specification, TimedPartialOrder)


def test_add_global_constraint():
    specification: Specification = TimedPartialOrder()
    node: int = 1
    lb: float = 10
    ub: float = 20
    specification.add_global_constraint(node, lb, ub)
    nodes = list(specification.nodes())
    n = nodes[0]
    assert n == node

    # Indices: 0=Node, 1=attributes
    nodes = specification.nodes(data=True)
    assert "lb" in nodes[n]
    assert "ub" in nodes[n]
    assert nodes[n]["lb"] == lb
    assert nodes[n]["ub"] == ub


def test_add_local_constraint():
    specification: Specification = TimedPartialOrder()
    src_node: int = 1
    tgt_node: int = 2
    lb: float = 10
    ub: float = 20
    specification.add_local_constraint(src_node, tgt_node, lb, ub)

    assert set(specification.nodes()) == set([src_node, tgt_node])
    assert list(specification.edges()) == [(src_node, tgt_node)]
    assert "lb" in specification[src_node][tgt_node]
    assert "ub" in specification[src_node][tgt_node]

    assert specification[src_node][tgt_node]["lb"] == lb
    assert specification[src_node][tgt_node]["ub"] == ub


def test_transitive_reduction():
    specification: Specification = TimedPartialOrder()
    specification.add_local_constraint(1, 2, 0, 10)
    specification.add_local_constraint(2, 3, 0, 10)
    specification.add_local_constraint(1, 3, 0, 10)
    specification.transitive_reduction()

    assert (1, 3) not in specification.edges()


def test_satisfy():
    pass
