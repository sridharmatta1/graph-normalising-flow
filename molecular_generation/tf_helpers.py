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


def cartesian_graph(a):
    """Same as utils.py's cartesian_graph -- given at least 2 elements
    in a, generates the Cartesian product of all elements in the list.
    """
    tile_a = tf.expand_dims(
        tf.tile(tf.expand_dims(a[0], 1), [1, tf.shape(a[1])[0]]), 2)
    tile_b = tf.expand_dims(
        tf.tile(tf.expand_dims(a[1], 0), [tf.shape(a[0])[0], 1]), 2)
    cart = tf.concat([tile_a, tile_b], axis=2)
    cart = tf.reshape(cart, [-1, 2])
    for c in a[2:]:
        tile_c = tf.tile(tf.expand_dims(c, 1), [1, tf.shape(cart)[0]])
        tile_c = tf.expand_dims(tile_c, 2)
        tile_c = tf.reshape(tile_c, [-1, 1])
        cart = tf.tile(cart, [tf.shape(c)[0], 1])
        cart = tf.concat([tile_c, cart], axis=1)
    return cart


def permutations(a, times=2):
    """Same as utils.py's permutations -- shortcut for generating the
    Cartesian product of self, using indices so that we can work with a
    small number of elements initially.
    """
    options = tf.range(tf.shape(a)[0])
    indices = cartesian_graph([options for _ in range(times)])
    gathered = tf.gather(a, indices)
    return gathered


def senders_receivers(n_node):
    """Same as utils.py's senders_receivers -- builds a fully-connected
    (every node pair, including self) directed edge list per graph,
    batched. Used to give the GNF flow's s/t networks a fully-connected
    graph structure, matching the paper's design for when no known
    graph structure exists (true both for community/ego generation and
    for the flow operating on embeddings here, which never sees real
    molecule connectivity -- only Phase 2's frozen encoder does).
    """
    def body(i, n_node_lower, n_node_cum, output):
        n_node_upper = n_node_cum[i]
        output = output.write(
            i, permutations(tf.range(n_node_lower, n_node_upper)))
        return (i + 1, n_node_cum[i], n_node_cum, output)

    num_graphs = tf.shape(n_node)[0]
    loop_condition = lambda i, *_: tf.less(i, num_graphs)
    initial_loop_vars = [
        0, 0,
        tf.cumsum(n_node),
        tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
    ]
    _, _, _, output = tf.while_loop(loop_condition,
                                    body,
                                    initial_loop_vars,
                                    back_prop=False)
    output = output.concat()
    return output[..., 0], output[..., 1]
