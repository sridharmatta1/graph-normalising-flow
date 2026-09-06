#!/bin/bash
#SBATCH --job-name=GNF_molecular_ae_v5
#SBATCH --output=logs/molecular_ae_v5_output.log
#SBATCH --error=logs/molecular_ae_v5_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 2, pushed further again: bond-type-aware attention
# (bond_aware_attention.py), isolated against v2's baseline --
# NOT v3's or v4's settings, both of which already regressed for
# unrelated reasons, so any change in results here can be
# attributed to this one change.
#
# Background: gnn.py's dm_self_attn_gnn (DMSelfAttention) already
# restricts attention to real bonded neighbors -- confirmed by
# reading its _build line by line, not a hyperparameter guess.
# What it does NOT do is use the bond's TYPE (single/double/
# triple) for anything: attention weights and messages come
# purely from node embeddings, so a double bond and a single bond
# produce identical attention behavior today. This adds a learned
# per-bond-type bias to both attention logits and attended values,
# so bond type can actually shape message passing, targeting the
# same "right count, wrong position" errors v4's iterative
# refinement tried (and failed) to fix a different way.
#
# Same as v2 otherwise: bond_class_weight=5.0, num_processing_steps=6,
# node_embedding_dim=64, latent_dim=256, num_train_iters=50000.
#
# Writes to a new v5 logdir. Remember: check_autoencoder_roundtrip.py
# needs --use_bond_aware_attention (plus the same capacity flags)
# passed explicitly when restoring a v5 checkpoint.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular auto-encoder v5: bond-type-aware attention"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_autoencoder.py \
    --data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/full_autoencoder_v5 \
    --num_train_iters 50000 \
    --train_batch_size 32 \
    --bond_class_weight 5.0 \
    --num_processing_steps 6 \
    --node_embedding_dim 64 \
    --latent_dim 256 \
    --use_bond_aware_attention \
    --log_every_n_steps 100 \
    --eval_every_n_steps 500 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_autoencoder_v5_qm9"

echo "=============================================="
echo "Molecular auto-encoder v5 training complete"
echo "End time: $(date)"
echo "=============================================="
