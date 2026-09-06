#!/bin/bash
#SBATCH --job-name=GNF_molecular_ae_v4
#SBATCH --output=logs/molecular_ae_v4_output.log
#SBATCH --error=logs/molecular_ae_v4_error.log
#SBATCH --mail-user=bidaralli@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Same as run_molecular_autoencoder_v4.sh (matta's account),
# pointed at bidaralli's paths -- see that script's comments for
# why (iterative bond-logit refinement, isolated against v2's
# baseline rather than v3's failed settings). Run on both
# accounts in parallel as a hedge against STUD partition
# contention.
# ============================================================
PYTHON=/home/bidaralli/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/bidaralli/Graph-Normalising-flow

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
    --wandb_run_name "molecular_autoencoder_v4_qm9_bidaralli"

echo "=============================================="
echo "Molecular auto-encoder v4 training complete"
echo "End time: $(date)"
echo "=============================================="
