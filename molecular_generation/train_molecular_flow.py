"""Phase 5: trains the (unconditioned) GNF flow on Phase 3's extracted
molecular embeddings -- the molecular analogue of train_grevnet_with_data.py.
Uses molecular_flow.py's MolecularGNFBlock (grevnet.py's GNFBlock
structure, plus ActNorm + scale-clamping fixes already proven
necessary for this project's coupling-layer flows -- see that module's
docstring for why the plain GNFBlock wasn't stable here). N-conditioning
is deliberately deferred, matching the original project plan.

The flow never sees real molecules or Phase 2's encoder/decoder -- it
only ever operates on the frozen encoder's embedding vectors, learning
their distribution via a RealNVP-style coupling-layer flow with a flat
N(0,I) prior (matching community/ego's own baseline, not the
N-conditioned variant).

Deliberately does NOT wire up sampling/decoding in this script.
community/ego's decoder (pred_adj) is a parameter-free distance
function, so their flow-training script can cheaply also build a
sampling+decode subgraph in the same checkpoint. Our decoder
(molecular_gnn.py's atom_type_logits/bond_type_logits) is a full set of
*learned* weights from a separately-trained checkpoint (Phase 2) --
combining that frozen checkpoint with this newly-trained flow at
generation time needs its own careful multi-Saver setup, which belongs
in Phase 6 (generate_molecules.py), not bolted onto this training loop.

Untested against a real TF graph before reaching the cluster, same as
every other script in this project (TF1/graph_nets aren't installable
on the dev machine this was written on) -- expect a first-run fix or
two, same pattern as train_molecular_autoencoder.py's early runs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
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
import tensorflow_probability as tfp
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
tfd = tfp.distributions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gnn import dm_self_attn_gnn, make_mlp_model

from molecular_flow import MolecularGNFBlock
from tf_helpers import reset_sess, senders_receivers

warnings.filterwarnings("ignore")

# Data params.
flags.DEFINE_string(
    'train_data_dir', 'molecular_generation/data',
    'Directory holding the embeddings_<run>_<chunk>.p files from '
    'generate_molecular_embeddings.py.')
flags.DEFINE_integer(
    'node_embedding_dim', 64,
    'Must match Phase 2\'s --node_embedding_dim (the encoder checkpoint '
    'used to extract these embeddings).')

# GNF params.
flags.DEFINE_integer('num_coupling_layers', 12, '')
flags.DEFINE_integer('latent_dim', 512, 'Hidden width inside the s/t '
                     'coupling networks\' MLPs.')
flags.DEFINE_integer('num_layers', 3, '')
flags.DEFINE_integer('attn_kq_dim', 64, '')
flags.DEFINE_integer('attn_v_dim', 64, '')
flags.DEFINE_integer('attn_num_heads', 2, '')
flags.DEFINE_integer('attn_concat_heads_output_dim', 64, '')
flags.DEFINE_bool('weight_sharing', False, '')
flags.DEFINE_float(
    'max_log_scale', 2.0,
    'Bounds the affine coupling scale s via max_log_scale*tanh(s/'
    'max_log_scale) before exp(s), so exp(s) is capped regardless of '
    'what the s network outputs. The plain grevnet.py GNFBlock this '
    'was originally built on (no clamping, real BatchNorm) diverged to '
    'NaN within ~1300 iterations when trained on molecular embeddings '
    '-- log_det_jacobian grew without bound every logged step, the '
    'same signature this project already fixed for NConditionedGNFBlock '
    'via this exact clamping (see molecular_flow.py).')

# Training params.
flags.DEFINE_string('logdir', 'molecular_generation/test_runs/flow', '')
flags.DEFINE_integer('train_batch_size', 32, '')
flags.DEFINE_integer(
    'train_epochs', 40,
    'How many times to repeat the embeddings file list -- must be '
    'large enough that num_train_iters * train_batch_size doesn\'t '
    'exceed train_epochs * (total molecules across all files).')
flags.DEFINE_integer('num_train_iters', 100000, '')
flags.DEFINE_integer('log_every_n_steps', 100, '')
flags.DEFINE_integer('summary_every_n_steps', 25, '')
flags.DEFINE_integer('save_every_n_steps', 2000, '')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, '')
flags.DEFINE_integer('random_seed', 12345, '')

# Optimizer params.
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


class MolecularFlowDataset():
    """Cycles through Phase 3's (node_embeddings, n_node) pickle chunks --
    same mechanism as train_grevnet_with_data.py's GrevnetDatasetFixed,
    just filtered to the embeddings_*.p files (train_data_dir also holds
    qm9_{train,val,test}.p / qm9.csv, which aren't embeddings).
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
            self.node_embeddings = d[0]
            self.n_node = d[1]
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
        self.prev_graph_ind = new_ind
        self.prev_node_embedding_ind = self.n_node_cs[new_ind - 1]
        return node_embeddings, n_node


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


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

    edges, globals_, receivers, senders, n_edge = transform_example(
        n_node_placeholder)
    graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                         edges=edges,
                                         globals=globals_,
                                         receivers=receivers,
                                         senders=senders,
                                         n_node=n_node_placeholder,
                                         n_edge=n_edge)
    batch_n_node = tf.reduce_sum(n_node_placeholder)

    half_dim = FLAGS.node_embedding_dim // 2
    make_mlp_fn = partial(make_mlp_model, FLAGS.latent_dim, half_dim,
                          FLAGS.num_layers)
    attn_gnn_fn = partial(
        dm_self_attn_gnn,
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        make_mlp_fn=make_mlp_fn,
        num_heads=FLAGS.attn_num_heads,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        concat=True,
        residual=False,
        layer_norm=False,
        kq_dim_division=True)

    grevnet = MolecularGNFBlock(attn_gnn_fn, FLAGS.num_coupling_layers,
                                half_dim, FLAGS.weight_sharing,
                                FLAGS.max_log_scale)

    grevnet_reverse_output, log_det_jacobian = grevnet(graphs_tuple,
                                                       inverse=True)
    grevnet_output_norm = tf.norm(grevnet_reverse_output.nodes, axis=1)

    mvn = tfd.MultivariateNormalDiag(tf.zeros(FLAGS.node_embedding_dim),
                                     tf.ones(FLAGS.node_embedding_dim))
    log_prob_zs = tf.reduce_sum(mvn.log_prob(grevnet_reverse_output.nodes))
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
    merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = reset_sess(config)

    if WANDB_AVAILABLE:
        wandb.init(
            project=FLAGS.wandb_project,
            name=FLAGS.wandb_run_name if FLAGS.wandb_run_name
            else "molecular_flow_qm9",
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
        "batch_n_node": batch_n_node,
    }

    dataset_generator = MolecularFlowDataset(FLAGS.train_data_dir,
                                             FLAGS.train_batch_size,
                                             FLAGS.train_epochs)

    for iteration in range(FLAGS.num_train_iters + 1):
        node_embeddings, n_node = dataset_generator.train_batch()
        feed_dict = {
            node_embeddings_placeholder: node_embeddings,
            n_node_placeholder: n_node,
        }
        train_values = sess.run(values_map, feed_dict=feed_dict)

        if train_writer and (iteration % FLAGS.summary_every_n_steps == 0):
            train_writer.add_summary(train_values['merge'], iteration)

        if iteration % FLAGS.log_every_n_steps == 0:
            logger.info("*" * 80)
            logger.info("iteration {}".format(iteration))
            logger.info(
                "total_loss={:.4f} per_node_loss={:.4f} "
                "log_prob_zs={:.4f} log_det_jacobian={:.4f}".format(
                    train_values["total_loss"],
                    train_values["per_node_loss"],
                    train_values["log_prob_zs"],
                    train_values["log_det_jacobian"]))
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
                }, step=iteration)

        if iteration % FLAGS.save_every_n_steps == 0:
            saver.save(sess, os.path.join(FLAGS.logdir, 'checkpoints'),
                      global_step=global_step)

    logger.info("Training complete.")


if __name__ == '__main__':
    app.run(main)
