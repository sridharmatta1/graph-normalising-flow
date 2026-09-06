#!/bin/bash
#SBATCH --job-name=GNF_molecular_ae_v4
#SBATCH --output=logs/molecular_ae_v4_output.log
#SBATCH --error=logs/molecular_ae_v4_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 2, pushed further again: iterative bond-logit refinement
# on top of v2's baseline (NOT v3's -- v3 stacked a higher
# bond_class_weight (10.0) and longer training (80k iters) onto
# v2 and came back WORSE (100%/60% vs v2's 100%/65% valid/exact-
# match), so that direction is a dead end. This isolates the one
# new change (num_bond_refine_steps) against the last known-good
# config, so any improvement (or regression) can be attributed to
# refinement specifically, not confounded with other changes.
#
# v2's round-trip result: 100% valid / 65% exact match on a
# 20-molecule held-out sample, after class-weighted loss (5.0) +
# more capacity fixed most of what was fixable that way. Remaining
# errors were "right atom/bond count, wrong specific position"
# (e.g. a double bond shifted one position over) -- errors from
# treating every bond pair as independent, which refine_bond_logits
# (molecular_gnn.py) directly targets by letting bond decisions on
# the same atom inform each other over a few rounds.
#
# Same as v2 otherwise: bond_class_weight=5.0, num_processing_steps=6,
# node_embedding_dim=64, latent_dim=256, num_train_iters=50000.
#
# Writes to a new v4 logdir. Remember: check_autoencoder_roundtrip.py
# needs the SAME --node_embedding_dim/--latent_dim/
# --num_processing_steps/--num_bond_refine_steps flags passed
# explicitly when restoring a v4 checkpoint.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular auto-encoder v4: iterative bond-logit refinement"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_autoencoder.py \
    --data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/full_autoencoder_v4 \
    --num_train_iters 50000 \
    --train_batch_size 32 \
    --bond_class_weight 5.0 \
    --num_processing_steps 6 \
    --node_embedding_dim 64 \
    --latent_dim 256 \
    --num_bond_refine_steps 3 \
    --log_every_n_steps 100 \
    --eval_every_n_steps 500 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_autoencoder_v4_qm9"

echo "=============================================="
echo "Molecular auto-encoder v4 training complete"
echo "End time: $(date)"
echo "=============================================="
