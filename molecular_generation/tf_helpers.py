"""Tiny TF session helper, kept separate from utils.py because that
module also imports matplotlib.pyplot at the top level (for unrelated
plotting helpers), which has repeatedly broken on the cluster's Python
3.6 conda envs (libtiff/Pillow ABI mismatches after installing rdkit).
Scripts in this directory only ever need session creation, not plotting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def reset_sess(config=None):
    """Same as utils.py's reset_sess, minus the matplotlib dependency.

    Deliberately does NOT call tf.reset_default_graph() -- callers use
    this after the full graph (placeholders, losses, optimizer, ...) is
    already built, so resetting the graph here would wipe out
    everything just constructed.
    """
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess
