"""Same pipeline stage as train_grevnet_with_data.py -- trains the GNF on
pre-computed node embeddings (from run_gnn.py + generate_grevnet_training_data.py)
-- but with NConditionedGNFBlock + the conditional prior (mu(N), sigma(N))
instead of GNFBlock + a flat N(0, I) prior.

generate_graphs.py downstream is unaffected: it restores whatever tensors are
in the 'sample_pred_adj' / 'sample_log_prob' / 'sample_n_node' collections
generically, regardless of which prior produced them.
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

from absl import flags
import graph_nets as gn
import tensorflow as tf
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

from conditional_prior import (
    ConditionalPriorNetwork,
    conditional_prior_log_prob,
    sample_conditional_prior,
)
from n_conditioning import NEmbedding
from grevnet import NConditionedGNFBlock
from gnn import *
from graph_data import *
from loss import *
from utils import *

warnings.filterwarnings("ignore")

flags.DEFINE_string('gpu', '0', '')

# Dataset params.
flags.DEFINE_string('dataset', 'graph_rnn_grid', '')
flags.DEFINE_bool('overfit_dataset', False, '')
flags.DEFINE_integer('overfit_num_graphs', 1, '')
flags.DEFINE_string('overfit_size_map', '', '')

# Training params.
flags.DEFINE_integer('train_epochs', 6, '')
flags.DEFINE_integer('train_batch_size', 32, '')
flags.DEFINE_bool('variable_dataset', False, '')
flags.DEFINE_integer('max_nodes', 1500, '')
flags.DEFINE_string('train_data_dir', '', '')
flags.DEFINE_integer('write_graphs_every_n_steps', 1000, '')
flags.DEFINE_integer('write_graphs_min_iter', 1000, '')
flags.DEFINE_integer('sample_size', 32, '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_string('logdir', 'test_runs/test_grevnet_conditioned_fixed_encoder',
                    'Where to write training files.')
flags.DEFINE_integer('num_train_iters', 200000, '')
flags.DEFINE_integer('log_every_n_steps', 1, '')
flags.DEFINE_integer('summary_every_n_steps', 25, '')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, '')
flags.DEFINE_integer('save_every_n_steps', 500, '')
flags.DEFINE_string('wandb_project', 'graph-normalising-flow', 'W&B project name.')
flags.DEFINE_string('wandb_run_name', '', 'W&B run name (optional).')

# Optimizer params.
flags.DEFINE_string(
    'lr_type', 'constant',
    'Can be constant, fixed_decay, polynomial_decay, or schedule.')
flags.DEFINE_float('lr', 1e-04, 'Learning rate for optimizer.')
flags.DEFINE_integer('lr_fixed_decay_steps', 1000, '')
flags.DEFINE_float('lr_fixed_decay_rate', 0.99, '')
flags.DEFINE_bool('lr_fixed_decay_staircase', False, '')
flags.DEFINE_float('adam_beta1', 0.9, '')
flags.DEFINE_float('adam_beta2', 0.999, '')
flags.DEFINE_float('adam_epsilon', 1e-08, '')
flags.DEFINE_bool('clip_gradient_by_norm', False,
                  'Whether to use norm-based gradient clipping.')
flags.DEFINE_float('clip_gradient_norm', 10.0,
                   'Value for norm-based gradient clipping.')

# NConditionedGNFBlock params.
flags.DEFINE_integer('num_coupling_layers', 10, '')
flags.DEFINE_bool('weight_sharing', False, '')
flags.DEFINE_bool(
    'use_batch_norm', True,
    'Renormalize x0/x1 before each coupling sub-step (see GNFBlock). '
    'Without this, exp(s) compounds across stacked layers and reliably '
    'overflows to inf/nan.')
flags.DEFINE_integer(
    'latent_dim', 2048,
    'Hidden dim used inside each FiLM-conditioned s/t network.')
flags.DEFINE_integer('n_embed_dim', 32, 'Dimension of the N embedding.')
flags.DEFINE_float(
    'prior_kl_weight', 0.1,
    'Weight on KL(N(mu(N),sigma(N)^2) || N(0,1)), added to the loss to '
    'discourage the conditional prior from drifting arbitrarily far from '
    'a standard normal. Does not disable N-conditioning -- mu/sigma can '
    'still differ meaningfully across N, this just penalizes the *size* '
    'of that deviation unless it earns its keep in log-likelihood. '
    'Confirmed empirically: mu/gamma drifting far from their identity '
    'init (mu~0.45, gamma~0.6 by the end of a 100k-iteration run, vs '
    '~0 and ~1 early on) correlated with generated embeddings landing '
    '5-11x farther from real ones in L2 norm than they should. Set to 0 '
    'to disable.')
flags.DEFINE_float(
    'max_log_scale', 2.0,
    'Bounds the affine coupling scale s via max_log_scale*tanh(s/'
    'max_log_scale) before exp(s), so exp(s) is capped to roughly '
    '[exp(-max_log_scale), exp(max_log_scale)] per layer regardless of '
    'what the s network outputs. Unbounded s compounds multiplicatively '
    'across num_coupling_layers stacked layers -- confirmed empirically '
    'to still cause a ~4x real-vs-generated embedding scale mismatch '
    'even with batch norm and prior_kl_weight already fixed.')

# Node feature params.
flags.DEFINE_integer('node_embedding_dim', 200,
                     'Dimension of node embeddings.')

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
logdir_prefix = os.environ.get('MLPATH')
if not logdir_prefix:
    logdir_prefix = '.'
LOGDIR = os.path.join(logdir_prefix, FLAGS.logdir)
os.makedirs(LOGDIR, exist_ok=True)
GRAPHS_LOGDIR = os.path.join(LOGDIR, "generated_graphs")
os.makedirs(GRAPHS_LOGDIR, exist_ok=True)

np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})
handlers = [logging.StreamHandler(sys.stdout)]
handlers.append(logging.FileHandler(os.path.join(LOGDIR, 'OUTPUT_LOG')))
logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger("logger")

tf.random.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)


class GrevnetDatasetFixed():
    def __init__(self, train_data_dir, train_batch_size):
        self.files = os.listdir(train_data_dir) * FLAGS.train_epochs
        self.file_ind = 0
        self.prev_graph_ind = 0
        self.prev_node_embedding_ind = 0
        self.train_batch_size = train_batch_size
        self.train_data_dir = train_data_dir
        with open(os.path.join(train_data_dir, self.files[self.file_ind]),
                  'rb') as f:
            d = pickle.load(f)
            self.node_embeddings = d[0]
            self.n_node = d[1]
            self.n_node_cs = np.cumsum(self.n_node)

    def train_batch(self):
        new_ind = self.prev_graph_ind + self.train_batch_size
        if new_ind > len(self.n_node):
            self.file_ind += 1
            print("****" * 50)
            print("Reading next file")
            with open(
                    os.path.join(self.train_data_dir,
                                 self.files[self.file_ind]), 'rb') as f:

                d = pickle.load(f)
                self.node_embeddings = d[0]
                self.n_node = d[1]
                self.prev_graph_ind = 0
                self.prev_node_embedding_ind = 0
                self.n_node_cs = np.cumsum(self.n_node)
                new_ind = self.prev_graph_ind + self.train_batch_size
        node_embeddings = self.node_embeddings[self.prev_node_embedding_ind:
                                               self.n_node_cs[new_ind - 1]]
        n_node = self.n_node[self.prev_graph_ind:new_ind]
        self.prev_graph_ind = new_ind
        self.prev_node_embedding_ind = self.n_node_cs[new_ind - 1]
        return node_embeddings, n_node


class GrevnetDatasetVariable():
    def __init__(self, train_data_dir, max_nodes):
        self.files = os.listdir(train_data_dir)
        self.file_ind = 0
        self.graph_ind = 0
        self.prev_graph_ind = 0
        self.prev_node_embedding_ind = 0
        self.max_nodes = max_nodes
        self.train_data_dir = train_data_dir
        with open(os.path.join(self.train_data_dir, self.files[self.file_ind]),
                  'rb') as f:
            d = pickle.load(f)
            self.node_embeddings = d[0]
            self.n_node = d[1]
            self.n_node_cs = np.cumsum(self.n_node)

    def train_batch(self):
        total_nodes = 0
        while True:
            if self.graph_ind >= len(self.n_node):
                node_embeddings = self.node_embeddings[
                    self.prev_node_embedding_ind:self.n_node_cs[self.graph_ind
                                                                - 1]]
                n_node = self.n_node[self.prev_graph_ind:self.graph_ind]
                self.file_ind += 1
                self.prev_graph_ind = 0
                self.graph_ind = 0
                self.prev_node_embedding_ind = 0
                filename = os.path.join(self.train_data_dir,
                                        self.files[self.file_ind])
                print("****" * 50)
                print("Reading next file {}".format(filename))
                with open(filename, 'rb') as f:
                    d = pickle.load(f)
                    self.node_embeddings = d[0]
                    self.n_node = d[1]
                    self.n_node_cs = np.cumsum(self.n_node)
                return node_embeddings, n_node
            if total_nodes + self.n_node[self.graph_ind] < self.max_nodes:
                total_nodes += self.n_node[self.graph_ind]
                self.graph_ind += 1
            else:
                break
        node_embeddings = self.node_embeddings[self.prev_node_embedding_ind:
                                               self.n_node_cs[self.graph_ind -
                                                              1]]
        n_node = self.n_node[self.prev_graph_ind:self.graph_ind]
        self.prev_graph_ind = self.graph_ind
        self.prev_node_embedding_ind = self.n_node_cs[self.graph_ind - 1]
        return node_embeddings, n_node


def transform_example(n_node):
    globals = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals, receivers, senders, n_edge


sizes = [int(x) for x in FLAGS.overfit_size_map.split(",")
         ] if FLAGS.overfit_size_map else None
dataset = OverfitGraphDataset(
    FLAGS.dataset, FLAGS.overfit_num_graphs, FLAGS.train_batch_size,
    FLAGS.node_embedding_dim,
    sizes) if FLAGS.overfit_dataset else GraphDataset(FLAGS.dataset,
                                                      FLAGS.node_embedding_dim)
node_embeddings_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, FLAGS.node_embedding_dim],
    name='node_embeddings_placeholder')
n_node_placeholder = tf.placeholder(dtype=tf.int32,
                                    shape=[FLAGS.train_batch_size],
                                    name='n_node_placeholder')

# Define GNN and output.
edges, globals, receivers, senders, n_edge = transform_example(
    n_node_placeholder)
graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                     edges=edges,
                                     globals=globals,
                                     receivers=receivers,
                                     senders=senders,
                                     n_node=n_node_placeholder,
                                     n_edge=n_edge)
batch_n_node = tf.reduce_sum(n_node_placeholder)

HALF_DIM = FLAGS.node_embedding_dim // 2
grevnet = NConditionedGNFBlock(
    num_timesteps=FLAGS.num_coupling_layers,
    node_embedding_dim=HALF_DIM,
    hidden_dim=FLAGS.latent_dim,
    n_embed_dim=FLAGS.n_embed_dim,
    weight_sharing=FLAGS.weight_sharing,
    use_batch_norm=FLAGS.use_batch_norm,
    max_log_scale=FLAGS.max_log_scale)

prior_n_embedding_mod = NEmbedding(FLAGS.n_embed_dim)
prior_net = ConditionalPriorNetwork(FLAGS.node_embedding_dim)

grevnet_reverse_output, log_det_jacobian = grevnet(graphs_tuple, inverse=True)
grevnet_output_norm = tf.norm(grevnet_reverse_output.nodes, axis=1)

train_n_embedding = prior_n_embedding_mod(graphs_tuple)
train_mu, train_sigma = prior_net(train_n_embedding)
log_prob_zs = tf.reduce_sum(
    conditional_prior_log_prob(grevnet_reverse_output.nodes,
                               graphs_tuple.n_node, train_mu, train_sigma))
log_prob_xs = log_prob_zs + log_det_jacobian

# KL(N(mu,sigma^2) || N(0,1)), summed over dims and graphs. Standard
# closed form; see the prior_kl_weight flag docstring for why this is
# here -- keeps the prior from drifting arbitrarily far from a
# well-behaved reference point without disabling N-conditioning itself.
prior_kl = tf.reduce_sum(0.5 * (tf.square(train_sigma) +
                                tf.square(train_mu) - 1.0 -
                                2.0 * tf.log(train_sigma)))

total_loss = -1 * log_prob_xs + FLAGS.prior_kl_weight * prior_kl
per_node_loss = total_loss / tf.cast(tf.reduce_sum(graphs_tuple.n_node),
                                     tf.float32)
# Optimizer.
global_step = tf.Variable(0, trainable=False, name='global_step')
lr = None
if FLAGS.lr_type == 'constant':
    lr = FLAGS.lr
elif FLAGS.lr_type == 'fixed_decay':
    lr = tf.train.exponential_decay(learning_rate=FLAGS.lr,
                                    global_step=global_step,
                                    decay_steps=FLAGS.lr_fixed_decay_steps,
                                    decay_rate=FLAGS.lr_fixed_decay_rate,
                                    staircase=FLAGS.lr_fixed_decay_staircase)
elif FLAGS.lr_type == 'polynomial_decay':
    lr = tf.train.polynomial_decay(learning_rate=FLAGS.lr,
                                   global_step=global_step,
                                   decay_steps=FLAGS.num_train_iters,
                                   end_learning_rate=FLAGS.lr / 100,
                                   power=0.5)
optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                   beta1=FLAGS.adam_beta1,
                                   beta2=FLAGS.adam_beta2,
                                   epsilon=FLAGS.adam_epsilon)
grads_and_vars = optimizer.compute_gradients(per_node_loss)
if FLAGS.clip_gradient_by_norm:
    grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.clip_gradient_norm), var)
                      for grad, var in grads_and_vars]
step_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Sample model: N-conditioned prior, sampled for sample_n_node_placeholder's
# N rather than whatever N showed up in a training batch. prior_n_embedding_mod
# and prior_net are reused (same weights) from the training pass above --
# only the N being conditioned on differs.
sample_n_node_placeholder = tf.placeholder(tf.int32,
                                           shape=[FLAGS.sample_size],
                                           name="sample_n_node_placeholder")
sample_edges, sample_globals, sample_receivers, sample_senders, sample_n_edge = transform_example(
    sample_n_node_placeholder)
dummy_sample_nodes = tf.zeros(
    [tf.reduce_sum(sample_n_node_placeholder), FLAGS.node_embedding_dim])
sample_graphs_tuple_structure = gn.graphs.GraphsTuple(
    nodes=dummy_sample_nodes,
    edges=sample_edges,
    globals=sample_globals,
    receivers=sample_receivers,
    senders=sample_senders,
    n_node=sample_n_node_placeholder,
    n_edge=sample_n_edge)
sample_n_embedding = prior_n_embedding_mod(sample_graphs_tuple_structure)
sample_mu, sample_sigma = prior_net(sample_n_embedding)
sample_nodes = sample_conditional_prior(sample_n_node_placeholder, sample_mu,
                                        sample_sigma)
sample_log_prob = conditional_prior_log_prob(sample_nodes,
                                             sample_n_node_placeholder,
                                             sample_mu, sample_sigma,
                                             per_node=True)
sample_graphs_tuple = sample_graphs_tuple_structure.replace(
    nodes=sample_nodes)

sample_grevnet_top = grevnet(sample_graphs_tuple, inverse=False)
sample_pred_adj = pred_adj(sample_grevnet_top,
                           distance_fn=scaled_hacky_sigmoid_l2)

tf.add_to_collection('sample_pred_adj', sample_pred_adj)
tf.add_to_collection('sample_log_prob', sample_log_prob)
tf.add_to_collection('sample_n_node', sample_n_node_placeholder)

tf.summary.scalar('total_loss', total_loss)
tf.summary.scalar('per_node_loss', per_node_loss)
tf.summary.scalar('log_prob_xs', log_prob_xs)
tf.summary.scalar('log_prob_zs', log_prob_zs)
tf.summary.scalar('log_det_jacobian', log_det_jacobian)
tf.summary.scalar('prior_kl', prior_kl)
tf.summary.scalar('prior_mu_norm', tf.norm(train_mu))
tf.summary.scalar('prior_sigma_mean', tf.reduce_mean(train_sigma))

merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = reset_sess(config)

train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'train'), sess.graph)

flags_map = tf.app.flags.FLAGS.flag_values_dict()
with open(os.path.join(LOGDIR, 'desc.txt'), 'w') as f:
    for (k, v) in flags_map.items():
        f.write("{}: {}\n".format(k, str(v)))

saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)

if WANDB_AVAILABLE:
    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.wandb_run_name if FLAGS.wandb_run_name else "grevnet_conditioned_{}".format(FLAGS.dataset),
        config=tf.app.flags.FLAGS.flag_values_dict()
    )

values_map = {
    "merge": merged,
    "step_op": step_op,
    "total_loss": total_loss,
    "per_node_loss": per_node_loss,
    "log_prob_zs": log_prob_zs,
    "log_prob_xs": log_prob_xs,
    "log_det_jacobian": log_det_jacobian,
    "prior_kl": prior_kl,
    "prior_mu_norm": tf.norm(train_mu),
    "prior_sigma_mean": tf.reduce_mean(train_sigma),
    "graphs_tuple": graphs_tuple,
    "batch_n_node": batch_n_node,
}

samples_map = {
    "sample_pred_adj": sample_pred_adj,
    "sample_grevnet_top": sample_grevnet_top,
    "sample_log_prob": sample_log_prob,
    "sample_grevnet_top_nodes": sample_grevnet_top.nodes,
    "sample_nodes": sample_nodes,
    "sample_n_node": sample_n_node_placeholder,
}

dataset_generator = None
if FLAGS.variable_dataset:
    dataset_generator = GrevnetDatasetVariable(
        os.path.join(logdir_prefix, FLAGS.train_data_dir), FLAGS.max_nodes)
else:
    dataset_generator = GrevnetDatasetFixed(
        os.path.join(logdir_prefix, FLAGS.train_data_dir),
        FLAGS.train_batch_size)

for iteration in range(0, FLAGS.num_train_iters + 1):
    node_embeddings, n_node = dataset_generator.train_batch()
    feed_dict = {
        node_embeddings_placeholder: node_embeddings,
        n_node_placeholder: n_node
    }
    train_values = sess.run(values_map, feed_dict=feed_dict)
    if train_writer and (iteration % FLAGS.summary_every_n_steps == 0):
        train_writer.add_summary(train_values['merge'], iteration)
    if iteration % FLAGS.log_every_n_steps == 0:
        logger.info("*" * 100)
        logger.info("iteration num: {}".format(iteration))
        logger.info("total loss: {}".format(train_values["total_loss"]))
        logger.info("per node loss: {}".format(train_values["per_node_loss"]))
        logger.info("log prob zs: {}".format(train_values["log_prob_zs"]))
        logger.info("log det jacobian: {}".format(
            train_values["log_det_jacobian"]))
        logger.info("prior kl: {}  prior mu norm: {}  prior sigma mean: {}".format(
            train_values["prior_kl"], train_values["prior_mu_norm"],
            train_values["prior_sigma_mean"]))
        logger.info("batch n_node {}, tot node {}, batch size {}".format(
            train_values["graphs_tuple"].n_node, train_values["batch_n_node"],
            len(train_values["graphs_tuple"].n_node)))
        if WANDB_AVAILABLE:
            wandb.log({
                "train/total_loss": float(train_values["total_loss"]),
                "train/per_node_loss": float(train_values["per_node_loss"]),
                "train/log_prob_zs": float(train_values["log_prob_zs"]),
                "train/log_prob_xs": float(train_values["log_prob_xs"]),
                "train/log_det_jacobian": float(train_values["log_det_jacobian"]),
            }, step=iteration)

    # Save model.
    if iteration % FLAGS.save_every_n_steps == 0:
        saver.save(sess,
                   os.path.join(LOGDIR, 'checkpoints'),
                   global_step=global_step)

    # Write out graphs.
    if iteration % FLAGS.write_graphs_every_n_steps == 0 and iteration > FLAGS.write_graphs_min_iter:
        graphs_dir = os.path.join(GRAPHS_LOGDIR, "iter_{}".format(iteration))
        os.makedirs(graphs_dir, exist_ok=True)
        feed_dict = {
            sample_n_node_placeholder:
            random.sample(FLAGS.sample_size * dataset.test_n_nodes(),
                          FLAGS.sample_size)
        }
        logger.info("*" * 100)
        logger.info("iteration num: {}".format(iteration))
        print("writing graphs...")
        graphs = []
        values = sess.run(samples_map, feed_dict=feed_dict)
        n_node = values["sample_grevnet_top"].n_node
        sample_log_prob_vals = values["sample_log_prob"]
        pred_adj_vals = values["sample_pred_adj"]
        adjacency = np.where(pred_adj_vals > 0.5, np.ones_like(pred_adj_vals),
                             np.zeros_like(pred_adj_vals))
        n_node_cum = np.cumsum(n_node)
        start_ind = 0
        for i in range(FLAGS.sample_size):
            end_ind = n_node_cum[i]
            num_nodes = end_ind - start_ind
            graph = adjacency[start_ind:end_ind, start_ind:end_ind]
            graph = nx.from_numpy_array(graph)
            single_sample_log_prob = np.mean(
                sample_log_prob_vals[start_ind:end_ind])
            visualize_graph(graph,
                            filename=os.path.join(
                                graphs_dir,
                                "graph_{}_prob_{:.2f}_nnode_{}.png".format(
                                    i, single_sample_log_prob, num_nodes)))
            graphs.append(graph)
            start_ind = end_ind
        pickle.dump(
            graphs,
            open(os.path.join(graphs_dir, "pickled.p".format(iteration)),
                 'wb'))
        logger.info("done writing graphs")
