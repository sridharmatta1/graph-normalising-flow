#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow
#SBATCH --output=logs/molecular_flow_output.log
#SBATCH --error=logs/molecular_flow_error.log
#SBATCH --mail-user=bidaralli@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Same as run_molecular_flow_full.sh (matta's account), pointed
# at bidaralli's paths -- see that script's comments for the
# max_log_scale=0.25 diagnosis. Run on both accounts in parallel
# as a hedge against STUD partition contention.
#
# Uses the Phase 2 v5 checkpoint copied over from matta's account
# (molecular_generation/checkpoints/phase2_final/) -- bidaralli's
# account never trained its own Phase 2 autoencoder.
# ============================================================
PYTHON=/home/bidaralli/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/bidaralli/Graph-Normalising-flow

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
    --max_log_scale 0.25 \
    --num_train_iters 100000 \
    --train_batch_size 32 \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_full_qm9_bidaralli"

echo "=============================================="
echo "Molecular GNF flow training complete"
echo "End time: $(date)"
echo "=============================================="
