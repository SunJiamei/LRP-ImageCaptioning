# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import keras.layers

from . import base


__all__ = [
    "dot",

    # todo: check why this makes problems to wrapper implementation.
    "skip_connection",
]


###############################################################################
###############################################################################
###############################################################################


def dot():
    input_shape = [None, 2]
    output_n = 1

    net = {}
    net["in"] = base.input_layer(shape=input_shape)
    net["out"] = base.dense_layer(net["in"], units=output_n,
                                  activation="linear")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })

    return net


def skip_connection():
    input_shape = [None, 1]
    output_n = 1

    net = {}
    net["in"] = base.input_layer(shape=input_shape)
    dense = keras.layers.Dense(units=output_n,
                               activation="linear",
                               use_bias=False)
    net["out"] = keras.layers.Add()([net["in"], dense(net["in"])])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })

    return net
