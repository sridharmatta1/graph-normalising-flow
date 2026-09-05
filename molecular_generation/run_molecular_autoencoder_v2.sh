#!/bin/bash
#SBATCH --job-name=GNF_molecular_ae_v2
#SBATCH --output=logs/molecular_ae_v2_output.log
#SBATCH --error=logs/molecular_ae_v2_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 2, pushed further: full-dataset training with a
# class-weighted bond loss + more model capacity, on top of
# v1's result (run_molecular_autoencoder_full.sh):
#   eval bond_accuracy ~94-97%, and after adding valence-aware
#   constrained decoding (qm9_chem.py), round-trip checks hit
#   100% valid / 50% exact match on a 20-molecule held-out
#   sample.
#
# This run targets the remaining "valid-but-different" gap
# (right shape, wrong bond somewhere) rather than the validity
# gap (already solved by decoding, not training). Changes from
# v1:
#   - bond_class_weight=5.0: most atom pairs are unbonded, so
#     unweighted loss can reach high aggregate bond_accuracy
#     mostly by nailing the easy majority ("no bond") class
#     while still getting real bonds wrong -- exactly what
#     shows up as valid-but-different rather than exact matches.
#     Upweighting the 3 real-bond classes targets that directly.
#   - More capacity (num_processing_steps 4->6, node_embedding_dim
#     32->64, latent_dim 128->256): bond-type prediction is a much
#     harder N x N combinatorial problem than atom-type (which
#     already saturated at 100%); more message-passing depth may
#     help the model reason about global consistency.
#
# Writes to a new v2 logdir so v1's checkpoint stays available for
# comparison. Remember: check_autoencoder_roundtrip.py needs the
# SAME --node_embedding_dim/--latent_dim/--num_processing_steps
# flags passed explicitly when restoring a v2 checkpoint (its
# defaults still match v1).
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular auto-encoder v2: class-weighted loss + more capacity"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_autoencoder.py \
    --data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/full_autoencoder_v2 \
    --num_train_iters 50000 \
    --train_batch_size 32 \
    --bond_class_weight 5.0 \
    --num_processing_steps 6 \
    --node_embedding_dim 64 \
    --latent_dim 256 \
    --log_every_n_steps 100 \
    --eval_every_n_steps 500 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_autoencoder_v2_qm9"

echo "=============================================="
echo "Molecular auto-encoder v2 training complete"
echo "End time: $(date)"
echo "=============================================="
