#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_n_cond
#SBATCH --output=logs/molecular_flow_n_conditioned_output.log
#SBATCH --error=logs/molecular_flow_n_conditioned_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 4/5: full training of the N-conditioned GNF flow
# (NConditionedMolecularGNFBlock + conditional prior) on Phase
# 3's extracted molecular embeddings. Only run this after
# run_molecular_flow_n_conditioned_smoke.sh's 2000-iteration
# smoke test confirms stability (see that script's comments for
# what to check) -- same order this project already followed for
# the unconditioned flow (run_molecular_flow_full.sh).
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular GNF flow (N-conditioned): full training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    --train_data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/flow_n_conditioned \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --num_train_iters 100000 \
    --train_batch_size 32 \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_n_conditioned_qm9"

echo "=============================================="
echo "Molecular GNF flow (N-conditioned) training complete"
echo "End time: $(date)"
echo "=============================================="
