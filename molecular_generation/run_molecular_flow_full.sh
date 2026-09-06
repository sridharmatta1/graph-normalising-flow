#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow
#SBATCH --output=logs/molecular_flow_output.log
#SBATCH --error=logs/molecular_flow_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 5: full training of the (unconditioned) GNF flow on
# Phase 3's extracted molecular embeddings (105,632 molecules,
# from the frozen Phase 2 v5 encoder -- 100% valid, 90% exact
# match on held-out reconstruction).
#
# Only run this after a short interactive smoke test confirms
# train_molecular_flow.py actually runs against a real TF graph
# (it hasn't yet -- untested against real TF/graph_nets on the
# dev machine this was written on, same as every other script in
# this project). e.g.:
#   python3 molecular_generation/train_molecular_flow.py \
#     --num_train_iters 2000 --logdir molecular_generation/test_runs/flow_smoke_test
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular GNF flow: full training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow.py \
    --train_data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/flow_full \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --num_train_iters 100000 \
    --train_batch_size 32 \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_full_qm9"

echo "=============================================="
echo "Molecular GNF flow training complete"
echo "End time: $(date)"
echo "=============================================="
