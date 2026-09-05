#!/bin/bash
#SBATCH --job-name=GNF_molecular_ae
#SBATCH --output=logs/molecular_ae_output.log
#SBATCH --error=logs/molecular_ae_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Full-dataset training of the molecular graph auto-encoder
# (Phase 2), on all ~132k kept QM9 molecules from Phase 1.
#
# The 500-molecule smoke test (2000, then 15000 iterations)
# plateaued at ~91% bond accuracy / 55% valid reconstructions /
# 10% exact matches with NO improvement between 2000 and 15000
# iterations -- ruling out "just needs more training time" on
# that tiny set. This run tests whether the full dataset's much
# greater chemical diversity is what the bond-type classifier
# actually needs to generalize better, rather than more epochs
# on the same 500 molecules.
#
# Uses the gnf_molecular conda env (not gnf_2026) -- this script
# never touches matplotlib/utils.py, only rdkit + graph_nets/
# sonnet/TF, so the earlier libtiff/Pillow breakage doesn't
# affect it.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular auto-encoder: full QM9 dataset"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_autoencoder.py \
    --data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/full_autoencoder \
    --num_train_iters 50000 \
    --train_batch_size 32 \
    --log_every_n_steps 100 \
    --eval_every_n_steps 500 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_autoencoder_full_qm9"

echo "=============================================="
echo "Molecular auto-encoder training complete"
echo "End time: $(date)"
echo "=============================================="
