"""

Utils
=====

This module provides utility functions and classes for various operations.

"""

import heapq

import numpy as np


class MaxHeapObj(object):
    """Overrides the comparison, so you can create a max heap easily.

    Parameters
    ----------
    val : Any
        The value to be stored in the heap object.

    Attributes
    ----------
    val : Any
        The value stored in the heap object.

    Methods
    -------
    __lt__(self, other)
        Overrides the less than comparison operator.
    __eq__(self, other)
        Overrides the equality comparison operator.
    __str__(self)
        Returns a string representation of the heap object.

    """

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class MinHeap(object):
    """A nice class-based interface to the heapq library."""

    def __init__(self):
        self.h = []

    def heappush(self, x):
        """Pushes an element onto the heap.

        Parameters
        ----------
        x : Any
            The element to be pushed onto the heap.

        """
        heapq.heappush(self.h, x)

    def heappop(self):
        """Pops the smallest element from the heap.

        Returns
        -------
        Any
            The smallest element from the heap.

        """
        return heapq.heappop(self.h)

    def __getitem__(self, i):
        """Returns the element at the given index.

        Parameters
        ----------
        i : int
            The index of the element to be retrieved.

        Returns
        -------
        Any
            The element at the given index.

        """
        return self.h[i]

    def __len__(self):
        """Returns the number of elements in the heap.

        Returns
        -------
        int
            The number of elements in the heap.

        """
        return len(self.h)


class MaxHeap(MinHeap):
    """
    A nice class-based interface to create a max heap, using the heapq library.

    """

    def heappush(self, x):
        """Pushes an element onto the max heap.

        Parameters
        ----------
        x : Any
            The element to be pushed onto the max heap.

        """
        heapq.heappush(self.h, MaxHeapObj(x))

    def heappop(self):
        """Pops the largest element from the max heap.

        Returns
        -------
        Any
            The largest element from the max heap.

        """
        return heapq.heappop(self.h).val

    def __getitem__(self, i):
        """Returns the element at the given index.

        Parameters
        ----------
        i : int
            The index of the element to be retrieved.

        Returns
        -------
        Any
            The element at the given index.

        """
        return self.h[i].val


def logx(x, base=2):
    """Calculates the logarithm of a number with a specified base.

    Parameters
    ----------
    x : float or array_like
        The number(s) for which the logarithm is to be calculated.
    base : float, optional
        The base of the logarithm. Default is 2.

    Returns
    -------
    float or array_like
        The logarithm(s) of the input number(s).

    """
    return np.asscalar(np.log(x) / np.log(base))


def xlogx(x, **kwargs):
    """Calculates the product of a number and its logarithm with a specified base.

    Parameters
    ----------
    x : float or array_like
        The number(s) for which the product is to be calculated.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `logx` function.

    Returns
    -------
    float or array_like
        The product(s) of the input number(s) and their logarithm(s).

    """
    return ylogx(x, x, **kwargs)


def xlogy(x, y, **kwargs):
    """Calculates the product of two numbers, one of which is multiplied by the logarithm of the other.

    Parameters
    ----------
    x : float or array_like
        The first number(s) for which the product is to be calculated.
    y : float or array_like
        The second number(s) for which the logarithm is to be calculated.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `logx` function.

    Returns
    -------
    float or array_like
        The product(s) of the input numbers, with one of them multiplied by their logarithm(s).

    """
    if isinstance(x, float) and x == 0.0:
        return 0.0
    if isinstance(x, int) and x == 0:
        return 0

    return x * logx(y, **kwargs)


def ylogx(x, y, **kwargs):
    """Calculates the product of two numbers, one of which is multiplied by the logarithm of the other.

    Parameters
    ----------
    x : float or array_like
        The first number(s) for which the logarithm is to be calculated.
    y : float or array_like
        The second number(s) for which the product is to be calculated.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `logx` function.

    Returns
    -------
    float or array_like
        The product(s) of the input numbers, with one of them multiplied by their logarithm(s).

    """
    if isinstance(y, float) and y == 0.0:
        return 0.0
    if isinstance(y, int) and y == 0:
        return 0
