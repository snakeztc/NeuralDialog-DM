import numpy as np


def vec2id(x, limits):
    """
    :param x: A discrete (multidimensional) quantity (often the state vector)
    :param limits: The limits of the discrete quantity (often statespace_limits)

    Returns a unique id by determining the number of possible values of ``x``
    that lie within ``limits``, and then seeing where this particular value of
    ``x` falls in that spectrum.

    .. note::

        See :py:meth:`~rlpy.Tools.GeneralTools.id2vec`, the inverse function.

    .. warning::

        This function assumes that (elements of) ``x`` takes integer values,
        and that ``limits`` are the lower and upper bounds on ``x``.
    """
    if isinstance(x, int):
        return x
    _id = 0
    for d in xrange(len(x) - 1, -1, -1):
        _id *= limits[d]
        _id += x[d]

    return _id