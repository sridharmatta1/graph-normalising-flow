#!/bin/bash
#SBATCH --job-name=GNF_molecular_ae_v3
#SBATCH --output=logs/molecular_ae_v3_output.log
#SBATCH --error=logs/molecular_ae_v3_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 2, pushed further again: dials up v2's two changes
# rather than a new architecture change, to see if there's more
# headroom in the current approach before committing to a bigger
# redesign (autoregressive/sequential bond decoding).
#
# v2 result (run_molecular_autoencoder_v2.sh): valence-aware
# round-trip hit 100% valid / 65% exact match on a 20-molecule
# held-out sample -- up from v1's 100%/50%. Remaining errors were
# mostly "right atom/bond count, wrong specific position" (e.g. a
# double bond shifted one position over) rather than validity
# failures.
#
# Changes from v2:
#   - bond_class_weight 5.0 -> 10.0: v2 already showed this
#     direction helps (bumping real-bond weight moved exact-match
#     50%->65%); testing whether more of the same continues to
#     help or has already saturated.
#   - num_train_iters 50000 -> 80000: v1's eval curve hadn't
#     clearly plateaued by 50k iterations; more training budget
#     tests whether that's still true for v2's larger model.
#   - Same architecture as v2 otherwise (num_processing_steps=6,
#     node_embedding_dim=64, latent_dim=256).
#
# Writes to a new v3 logdir so v1/v2 checkpoints stay available
# for comparison. Remember: check_autoencoder_roundtrip.py needs
# the SAME --node_embedding_dim/--latent_dim/--num_processing_steps
# flags passed explicitly when restoring a v3 checkpoint.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular auto-encoder v3: higher bond_class_weight + longer training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_autoencoder.py \
    --data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/full_autoencoder_v3 \
    --num_train_iters 80000 \
    --train_batch_size 32 \
    --bond_class_weight 10.0 \
    --num_processing_steps 6 \
    --node_embedding_dim 64 \
    --latent_dim 256 \
    --log_every_n_steps 100 \
    --eval_every_n_steps 500 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_autoencoder_v3_qm9"

echo "=============================================="
echo "Molecular auto-encoder v3 training complete"
echo "End time: $(date)"
echo "=============================================="
