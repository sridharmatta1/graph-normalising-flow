"""Phase 2, part 3: trains the molecular graph auto-encoder -- QM9's
analogue of run_gnn.py, which trains the structural community/ego
auto-encoder. Same overall recipe (encode a real graph, reconstruct it,
minimize reconstruction loss), but reconstructs atom types + bond types
(molecular_gnn.py's heads) instead of a single edge/no-edge adjacency
(gnn.py's pred_adj).

Reuses gnn.py's TimestepGNN + dm_self_attn_gnn (the actual message-passing
GNN) completely unchanged -- only the input embedding, decoder heads, and
loss (all in molecular_gnn.py) are new.

Run small first (--max_molecules), matching this project's established
pattern of validating cheaply before committing to a full run: this
script has never been run against the real TF graph before (TF1/
graph_nets aren't installable on the dev machine used to write it), so
the small-batch run here is this code's first real test, not just a
sanity check.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import logging
import os
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gnn import TimestepGNN, dm_self_attn_gnn, make_mlp_model
from utils import reset_sess

from molecular_gnn import embed_atom_features, molecular_reconstruction_loss
from qm9_graph_data import QM9GraphDataset

warnings.filterwarnings("ignore")

# Data params.
flags.DEFINE_string('data_dir', 'molecular_generation/data',
                    'Directory holding qm9_{train,val,test}.p from '
                    'preprocess_qm9.py.')
flags.DEFINE_integer(
    'max_molecules', 0,
    'If > 0, only use the first N training molecules (and N/10 val '
    'molecules) -- for small-batch validation before a full run.')

# Attention/GNN params (mirrors run_gnn.py's dm_attn defaults, but with a
# much smaller processing budget -- QM9 molecules have <= 9 atoms, vs.
# community/ego's up to 20, so they need far fewer message-passing steps
# to let information reach every node).
flags.DEFINE_integer('attn_kq_dim', 32, '')
flags.DEFINE_integer('attn_v_dim', 32, '')
flags.DEFINE_integer('attn_num_heads', 2, '')
flags.DEFINE_integer('attn_concat_heads_output_dim', 32, '')
flags.DEFINE_integer('num_processing_steps', 4, '')
flags.DEFINE_integer('node_embedding_dim', 32,
                     'Internal per-atom embedding width used throughout '
                     'the GNN -- unrelated to NUM_ATOM_TYPES, which is '
                     'the raw one-hot input dimension before embedding.')
flags.DEFINE_integer('latent_dim', 128, 'Hidden width inside every MLP.')
flags.DEFINE_integer('num_mlp_layers', 2, '')
flags.DEFINE_bool('weight_sharing', True, '')
flags.DEFINE_bool('use_batch_norm', False,
                  'QM9 batches are tiny (<= a few hundred atoms per '
                  'batch) -- batch norm statistics are noisier here than '
                  'for community/ego, so this defaults off; can be '
                  'revisited once the basic pipeline is validated.')
flags.DEFINE_bool('residual', False, '')

# Training params.
flags.DEFINE_string('logdir', 'molecular_generation/test_runs/autoencoder',
                    '')
flags.DEFINE_integer('train_batch_size', 32, '')
flags.DEFINE_integer('num_train_iters', 2000, '')
flags.DEFINE_integer('log_every_n_steps', 20, '')
flags.DEFINE_integer('eval_every_n_steps', 100, '')
flags.DEFINE_integer('save_every_n_steps', 500, '')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_float('lr', 1e-03, '')
flags.DEFINE_string('wandb_project', 'graph-normalising-flow', '')
flags.DEFINE_string('wandb_run_name', '', '')

FLAGS = flags.FLAGS


def main(argv):
    del argv
    os.makedirs(FLAGS.logdir, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
               logging.FileHandler(os.path.join(FLAGS.logdir, 'OUTPUT_LOG'))]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger = logging.getLogger("logger")

    tf.random.set_random_seed(FLAGS.random_seed)

    dataset = QM9GraphDataset(FLAGS.data_dir, max_molecules=FLAGS.max_molecules)
    logger.info("Loaded {} train / {} val molecules".format(
        len(dataset.train_graphs), len(dataset.test_graphs)))

    raw_graph_phs = gn.utils_tf.placeholders_from_networkxs(
        dataset.train_graphs, force_dynamic_num_graphs=True,
        name="raw_graph_phs")
    raw_graph_phs.n_node.set_shape([FLAGS.train_batch_size])

    embedded_graph_phs = embed_atom_features(
        raw_graph_phs, FLAGS.latent_dim, FLAGS.node_embedding_dim,
        FLAGS.num_mlp_layers)

    make_mlp_fn = partial(make_mlp_model, FLAGS.latent_dim,
                          FLAGS.node_embedding_dim, FLAGS.num_mlp_layers)
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

    is_training = tf.placeholder(tf.bool, name="is_training")
    gnn = TimestepGNN(
        attn_gnn_fn,
        FLAGS.num_processing_steps,
        weight_sharing=FLAGS.weight_sharing,
        use_batch_norm=FLAGS.use_batch_norm,
        residual=FLAGS.residual,
        test_local_stats=True)
    gnn_output = gnn(embedded_graph_phs, is_training=is_training)

    losses = molecular_reconstruction_loss(
        raw_graph_phs, gnn_output, FLAGS.node_embedding_dim,
        FLAGS.latent_dim, FLAGS.num_mlp_layers)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        step_op = optimizer.minimize(losses['total_loss'],
                                     global_step=global_step)

    tf.summary.scalar('total_loss', losses['total_loss'])
    tf.summary.scalar('mean_loss', losses['mean_loss'])
    tf.summary.scalar('atom_loss', losses['atom_loss'])
    tf.summary.scalar('bond_loss', losses['bond_loss'])
    tf.summary.scalar('atom_accuracy', losses['atom_accuracy'])
    tf.summary.scalar('bond_accuracy', losses['bond_accuracy'])
    merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = reset_sess(config)

    if WANDB_AVAILABLE:
        wandb.init(
            project=FLAGS.wandb_project,
            name=FLAGS.wandb_run_name if FLAGS.wandb_run_name
            else "molecular_autoencoder_qm9",
            config=FLAGS.flag_values_dict())

    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'train'), sess.graph)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)

    values_map = {
        'merged': merged,
        'step_op': step_op,
        'total_loss': losses['total_loss'],
        'mean_loss': losses['mean_loss'],
        'atom_loss': losses['atom_loss'],
        'bond_loss': losses['bond_loss'],
        'atom_accuracy': losses['atom_accuracy'],
        'bond_accuracy': losses['bond_accuracy'],
    }
    eval_values_map = {
        'eval_total_loss': losses['total_loss'],
        'eval_atom_accuracy': losses['atom_accuracy'],
        'eval_bond_accuracy': losses['bond_accuracy'],
    }
    for k, v in values_map.items():
        if k not in ('merged', 'step_op'):
            tf.add_to_collection(k, v)
    tf.add_to_collection('gnn_output_nodes', gnn_output.nodes)

    for iteration in range(FLAGS.num_train_iters + 1):
        train_batch = dataset.get_next_train_batch(FLAGS.train_batch_size)
        feed_dict = {raw_graph_phs: train_batch, is_training: True}
        train_values = sess.run(values_map, feed_dict=feed_dict)

        if iteration % FLAGS.log_every_n_steps == 0:
            train_writer.add_summary(train_values['merged'], iteration)
            logger.info("*" * 80)
            logger.info("iteration {}".format(iteration))
            logger.info("total_loss={:.4f} atom_loss={:.4f} bond_loss={:.4f}"
                       .format(train_values['total_loss'],
                               train_values['atom_loss'],
                               train_values['bond_loss']))
            logger.info("atom_accuracy={:.4f} bond_accuracy={:.4f}".format(
                train_values['atom_accuracy'], train_values['bond_accuracy']))
            if WANDB_AVAILABLE:
                wandb.log({
                    "train/total_loss": float(train_values['total_loss']),
                    "train/atom_loss": float(train_values['atom_loss']),
                    "train/bond_loss": float(train_values['bond_loss']),
                    "train/atom_accuracy": float(train_values['atom_accuracy']),
                    "train/bond_accuracy": float(train_values['bond_accuracy']),
                }, step=iteration)

        if iteration % FLAGS.eval_every_n_steps == 0:
            test_batch = dataset.get_random_test_batch(FLAGS.train_batch_size)
            eval_values = sess.run(
                eval_values_map,
                feed_dict={raw_graph_phs: test_batch, is_training: False})
            logger.info("EVAL iteration {}: total_loss={:.4f} "
                       "atom_accuracy={:.4f} bond_accuracy={:.4f}".format(
                           iteration, eval_values['eval_total_loss'],
                           eval_values['eval_atom_accuracy'],
                           eval_values['eval_bond_accuracy']))
            if WANDB_AVAILABLE:
                wandb.log({
                    "eval/total_loss": float(eval_values['eval_total_loss']),
                    "eval/atom_accuracy": float(eval_values['eval_atom_accuracy']),
                    "eval/bond_accuracy": float(eval_values['eval_bond_accuracy']),
                }, step=iteration)

        if iteration % FLAGS.save_every_n_steps == 0:
            saver.save(sess, os.path.join(FLAGS.logdir, 'checkpoints'),
                      global_step=global_step)

    logger.info("Training complete.")


if __name__ == '__main__':
    tf.app.run(main)
