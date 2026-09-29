"""Phase 4 extension: trains the molecular flow conditioned JOINTLY on N
and a real chemical property (default: 'mu', dipole moment) -- the
property-conditioned counterpart of train_molecular_flow_n_conditioned.py.

Wires together:
  - NPropertyEmbedding (property_conditioning.py): N + property ->
    one embedding, in place of n_conditioning.py's plain NEmbedding.
  - NConditionedMolecularGNFBlock (molecular_flow.py), unchanged except
    for its new embedding_module parameter -- everything else (ActNorm,
    max_log_scale clamping, FiLM wiring) is identical to the N-only
    conditioned flow.
  - ConditionalPriorNetwork / conditional_prior_log_prob
    (conditional_prior.py), also unchanged -- it's agnostic to what
    embedding produced its input, so a joint N+property embedding
    slots in with no modification there either.

Reads embeddings_*.p chunks produced by
generate_molecular_embeddings.py --property_name=mu, which are 3-tuples
(node_embeddings, n_node, property_values) instead of the usual
2-tuple -- MUST be a separate directory from the plain N-conditioned
flow's embeddings (those are still 2-tuples with no property values).

Untested against a real TF graph before reaching the cluster, same as
every other script in this project -- expect a first-run fix or two.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import random
import sys
import warnings

from absl import app
from absl import flags
import graph_nets as gn
import numpy as np
import tensorflow as tf
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from conditional_prior import ConditionalPriorNetwork, conditional_prior_log_prob

from molecular_flow import NConditionedMolecularGNFBlock
from property_conditioning import NPropertyEmbedding
from tf_helpers import reset_sess, senders_receivers

warnings.filterwarnings("ignore")

# Data params.
flags.DEFINE_string(
    'train_data_dir', 'molecular_generation/data/property_embeddings',
    'Directory holding the PROPERTY-tagged embeddings_<run>_<chunk>.p '
    'files (3-tuples: node_embeddings, n_node, property_values) from '
    'generate_molecular_embeddings.py --property_name=<name>. Must NOT '
    'point at the plain N-only embeddings directory (2-tuples, no '
    'property values).')
flags.DEFINE_integer(
    'node_embedding_dim', 64,
    'Must match Phase 2\'s --node_embedding_dim (the encoder checkpoint '
    'used to extract these embeddings).')

# Property normalization -- computed once from qm9_train.p (see
# property_conditioning.py's NPropertyEmbedding docstring). Defaults
# are for 'mu' (dipole moment): mean=2.6520, std=1.4035.
flags.DEFINE_string('property_name', 'mu',
                    'Which QM9 property these embeddings were tagged '
                    'with -- must match --property_name from '
                    'generate_molecular_embeddings.py. Purely for '
                    'logging/bookkeeping here; the actual value is '
                    'already baked into the loaded embeddings\' '
                    'property_values array.')
flags.DEFINE_float('property_mean', 2.6520,
                   'Training-set mean of the conditioning property, '
                   'for normalization. Recompute if using a property '
                   'other than mu.')
flags.DEFINE_float('property_std', 1.4035,
                   'Training-set std of the conditioning property, '
                   'for normalization. Recompute if using a property '
                   'other than mu.')

# GNF params.
flags.DEFINE_integer('num_coupling_layers', 12, '')
flags.DEFINE_integer(
    'hidden_dim', 256,
    'Hidden width inside each FiLM-conditioned s/t network -- same '
    'default as train_molecular_flow_n_conditioned.py.')
flags.DEFINE_integer('n_embed_dim', 32,
                     'Dimension of the shared N+property embedding.')
flags.DEFINE_bool('weight_sharing', False, '')
flags.DEFINE_float('max_log_scale', 0.25, '')

# Training params.
flags.DEFINE_string(
    'logdir', 'molecular_generation/test_runs/flow_property_conditioned', '')
flags.DEFINE_integer('train_batch_size', 32, '')
flags.DEFINE_integer('train_epochs', 40, '')
flags.DEFINE_integer('num_train_iters', 100000, '')
flags.DEFINE_integer('log_every_n_steps', 100, '')
flags.DEFINE_integer('summary_every_n_steps', 25, '')
flags.DEFINE_integer('save_every_n_steps', 2000, '')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, '')
flags.DEFINE_integer('random_seed', 12345, '')

# Optimizer params -- same defaults as train_molecular_flow_n_conditioned.py.
flags.DEFINE_string('lr_type', 'fixed_decay', '')
flags.DEFINE_float('lr', 1e-04, '')
flags.DEFINE_integer('lr_fixed_decay_steps', 1000, '')
flags.DEFINE_float('lr_fixed_decay_rate', 0.99, '')
flags.DEFINE_bool('clip_gradient_by_norm', False, '')
flags.DEFINE_float('clip_gradient_norm', 10.0, '')
flags.DEFINE_float('adam_beta1', 0.9, '')
flags.DEFINE_float('adam_beta2', 0.999, '')
flags.DEFINE_float('adam_epsilon', 1e-08, '')

flags.DEFINE_string('wandb_project', 'graph-normalising-flow', '')
flags.DEFINE_string('wandb_run_name', '', '')

FLAGS = flags.FLAGS


class PropertyMolecularFlowDataset():
    """Same cycling logic as train_molecular_flow_n_conditioned.py's own
    MolecularFlowDataset, extended to also slice out each batch's
    property_values from the 3-tuple embeddings_*.p chunks. A
    standalone copy rather than a shared import, for the same reason
    that class is itself a standalone copy: importing across scripts
    that each define overlapping absl flags hits
    absl.flags.DuplicateFlagError.
    """

    def __init__(self, train_data_dir, train_batch_size, train_epochs):
        self.files = sorted(f for f in os.listdir(train_data_dir)
                            if f.startswith('embeddings_')) * train_epochs
        self.file_ind = 0
        self.prev_graph_ind = 0
        self.prev_node_embedding_ind = 0
        self.train_batch_size = train_batch_size
        self.train_data_dir = train_data_dir
        self._load_file(self.file_ind)

    def _load_file(self, file_ind):
        path = os.path.join(self.train_data_dir, self.files[file_ind])
        with open(path, 'rb') as f:
            d = pickle.load(f)
            if len(d) != 3:
                raise ValueError(
                    "Expected a 3-tuple (node_embeddings, n_node, "
                    "property_values) from {} -- got {} elements. Did "
                    "you point --train_data_dir at plain N-only "
                    "embeddings (2-tuples) by mistake?".format(
                        path, len(d)))
            self.node_embeddings = d[0]
            self.n_node = d[1]
            self.property_values = d[2]
            self.n_node_cs = np.cumsum(self.n_node)

    def train_batch(self):
        new_ind = self.prev_graph_ind + self.train_batch_size
        if new_ind > len(self.n_node):
            self.file_ind += 1
            print("Reading next embeddings file: {}".format(
                self.files[self.file_ind]))
            self._load_file(self.file_ind)
            self.prev_graph_ind = 0
            self.prev_node_embedding_ind = 0
            new_ind = self.prev_graph_ind + self.train_batch_size
        node_embeddings = self.node_embeddings[
            self.prev_node_embedding_ind:self.n_node_cs[new_ind - 1]]
        n_node = self.n_node[self.prev_graph_ind:new_ind]
        property_values = self.property_values[self.prev_graph_ind:new_ind]
        self.prev_graph_ind = new_ind
        self.prev_node_embedding_ind = self.n_node_cs[new_ind - 1]
        return node_embeddings, n_node, property_values


def transform_example(n_node):
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, receivers, senders, n_edge


def main(argv):
    del argv
    os.makedirs(FLAGS.logdir, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
               logging.FileHandler(os.path.join(FLAGS.logdir, 'OUTPUT_LOG'))]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger = logging.getLogger("logger")

    tf.random.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    node_embeddings_placeholder = tf.placeholder(
        dtype=tf.float32, shape=[None, FLAGS.node_embedding_dim],
        name='node_embeddings_placeholder')
    n_node_placeholder = tf.placeholder(dtype=tf.int32,
                                        shape=[FLAGS.train_batch_size],
                                        name='n_node_placeholder')
    property_placeholder = tf.placeholder(dtype=tf.float32,
                                          shape=[FLAGS.train_batch_size],
                                          name='property_placeholder')

    edges, receivers, senders, n_edge = transform_example(n_node_placeholder)
    graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                         edges=edges,
                                         globals=property_placeholder,
                                         receivers=receivers,
                                         senders=senders,
                                         n_node=n_node_placeholder,
                                         n_edge=n_edge)

    half_dim = FLAGS.node_embedding_dim // 2
    flow_embedding = NPropertyEmbedding(FLAGS.n_embed_dim,
                                       FLAGS.property_mean,
                                       FLAGS.property_std)
    grevnet = NConditionedMolecularGNFBlock(
        num_timesteps=FLAGS.num_coupling_layers,
        node_embedding_dim=half_dim,
        hidden_dim=FLAGS.hidden_dim,
        n_embed_dim=FLAGS.n_embed_dim,
        weight_sharing=FLAGS.weight_sharing,
        max_log_scale=FLAGS.max_log_scale,
        embedding_module=flow_embedding)

    prior_embedding = NPropertyEmbedding(FLAGS.n_embed_dim,
                                        FLAGS.property_mean,
                                        FLAGS.property_std)
    prior_net = ConditionalPriorNetwork(FLAGS.node_embedding_dim)

    grevnet_reverse_output, log_det_jacobian = grevnet(graphs_tuple,
                                                       inverse=True)
    grevnet_output_norm = tf.norm(grevnet_reverse_output.nodes, axis=1)

    prior_n_embedding = prior_embedding(graphs_tuple)
    mu, sigma = prior_net(prior_n_embedding)
    log_prob_zs = tf.reduce_sum(
        conditional_prior_log_prob(grevnet_reverse_output.nodes,
                                   graphs_tuple.n_node, mu, sigma))
    log_prob_xs = log_prob_zs + log_det_jacobian
    total_loss = -1 * log_prob_xs
    per_node_loss = total_loss / tf.cast(tf.reduce_sum(graphs_tuple.n_node),
                                         tf.float32)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr = None
    if FLAGS.lr_type == 'constant':
        lr = FLAGS.lr
    elif FLAGS.lr_type == 'fixed_decay':
        lr = tf.train.exponential_decay(
            learning_rate=FLAGS.lr, global_step=global_step,
            decay_steps=FLAGS.lr_fixed_decay_steps,
            decay_rate=FLAGS.lr_fixed_decay_rate)
    elif FLAGS.lr_type == 'polynomial_decay':
        lr = tf.train.polynomial_decay(
            learning_rate=FLAGS.lr, global_step=global_step,
            decay_steps=FLAGS.num_train_iters,
            end_learning_rate=FLAGS.lr / 100, power=0.5)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                       beta1=FLAGS.adam_beta1,
                                       beta2=FLAGS.adam_beta2,
                                       epsilon=FLAGS.adam_epsilon)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        grads_and_vars = optimizer.compute_gradients(per_node_loss)
        if FLAGS.clip_gradient_by_norm:
            grads_and_vars = [
                (tf.clip_by_norm(grad, FLAGS.clip_gradient_norm), var)
                for grad, var in grads_and_vars
            ]
        step_op = optimizer.apply_gradients(grads_and_vars,
                                            global_step=global_step)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('per_node_loss', per_node_loss)
    tf.summary.scalar('log_prob_xs', log_prob_xs)
    tf.summary.scalar('log_prob_zs', log_prob_zs)
    tf.summary.scalar('log_det_jacobian', log_det_jacobian)
    tf.summary.scalar('prior_sigma_mean', tf.reduce_mean(sigma))
    merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = reset_sess(config)

    if WANDB_AVAILABLE:
        wandb.init(
            project=FLAGS.wandb_project,
            name=FLAGS.wandb_run_name if FLAGS.wandb_run_name
            else "molecular_flow_property_conditioned_{}".format(
                FLAGS.property_name),
            config=FLAGS.flag_values_dict())

    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'train'), sess.graph)

    flags_map = FLAGS.flag_values_dict()
    with open(os.path.join(FLAGS.logdir, 'desc.txt'), 'w') as f:
        for (k, v) in flags_map.items():
            f.write("{}: {}\n".format(k, str(v)))

    saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)

    values_map = {
        "merge": merged,
        "step_op": step_op,
        "total_loss": total_loss,
        "per_node_loss": per_node_loss,
        "log_prob_zs": log_prob_zs,
        "log_prob_xs": log_prob_xs,
        "log_det_jacobian": log_det_jacobian,
        "grevnet_output_norm": grevnet_output_norm,
        "prior_sigma_mean": tf.reduce_mean(sigma),
    }

    dataset_generator = PropertyMolecularFlowDataset(FLAGS.train_data_dir,
                                                     FLAGS.train_batch_size,
                                                     FLAGS.train_epochs)

    for iteration in range(FLAGS.num_train_iters + 1):
        node_embeddings, n_node, property_values = dataset_generator.train_batch()
        feed_dict = {
            node_embeddings_placeholder: node_embeddings,
            n_node_placeholder: n_node,
            property_placeholder: property_values,
        }
        train_values = sess.run(values_map, feed_dict=feed_dict)

        if train_writer and (iteration % FLAGS.summary_every_n_steps == 0):
            train_writer.add_summary(train_values['merge'], iteration)

        if iteration % FLAGS.log_every_n_steps == 0:
            logger.info("*" * 80)
            logger.info("iteration {}".format(iteration))
            logger.info(
                "total_loss={:.4f} per_node_loss={:.4f} "
                "log_prob_zs={:.4f} log_det_jacobian={:.4f} "
                "prior_sigma_mean={:.4f}".format(
                    train_values["total_loss"],
                    train_values["per_node_loss"],
                    train_values["log_prob_zs"],
                    train_values["log_det_jacobian"],
                    train_values["prior_sigma_mean"]))
            logger.info("grevnet output norm (mean): {:.4f}".format(
                np.mean(train_values["grevnet_output_norm"])))
            if WANDB_AVAILABLE:
                wandb.log({
                    "train/total_loss": float(train_values["total_loss"]),
                    "train/per_node_loss": float(train_values["per_node_loss"]),
                    "train/log_prob_zs": float(train_values["log_prob_zs"]),
                    "train/log_prob_xs": float(train_values["log_prob_xs"]),
                    "train/log_det_jacobian": float(train_values["log_det_jacobian"]),
                    "train/output_norm": float(np.mean(train_values["grevnet_output_norm"])),
                    "train/prior_sigma_mean": float(train_values["prior_sigma_mean"]),
                }, step=iteration)

        if iteration % FLAGS.save_every_n_steps == 0:
            saver.save(sess, os.path.join(FLAGS.logdir, 'checkpoints'),
                      global_step=global_step)

    logger.info("Training complete.")


if __name__ == '__main__':
    app.run(main)
