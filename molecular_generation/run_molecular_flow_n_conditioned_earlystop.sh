#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_n_cond_es
#SBATCH --output=logs/molecular_flow_n_conditioned_earlystop_output.log
#SBATCH --error=logs/molecular_flow_n_conditioned_earlystop_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# N-conditioned molecular flow, retrained with a held-out
# validation loss and early stopping, instead of a fixed 100k
# iterations judged by the (noisy) training loss.
#
#   1. Encodes the VAL split (13,204 molecules) with the frozen
#      Phase 2 encoder -- skipped if already done. The test split
#      is never touched.
#   2. Trains the same architecture/hyperparameters as
#      run_molecular_flow_n_conditioned_full.sh, computing the
#      validation per-node loss every 2000 steps. Stops when it has
#      not improved by more than 0.05 for 10 evaluations in a row
#      (20,000 steps); num_train_iters is the hard cap.
#      The lowest-validation checkpoint is kept at
#      $FLOW_LOGDIR/best/best.
#   3. Generates 10,000 molecules from that best checkpoint
#      (target_n=0, sigma_scale=1.0), same protocol as the 5-seed
#      tables, so the result is directly comparable.
#
# Writes only to new directories/files -- nothing from earlier runs
# is overwritten.
# ============================================================
SEED=12345
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_LOGDIR=molecular_generation/test_runs/flow_n_conditioned_earlystop
VAL_DIR=molecular_generation/data/val_embeddings

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

if ls $VAL_DIR/embeddings_* >/dev/null 2>&1; then
    echo "Validation embeddings already present in $VAL_DIR, skipping extraction."
else
    echo "=============================================="
    echo "Encoding the VAL split with the frozen Phase 2 encoder"
    echo "Start time: $(date)"
    echo "=============================================="
    mkdir -p $VAL_DIR
    srun $PYTHON -u molecular_generation/generate_molecular_embeddings.py \
        --split val \
        --output_file $VAL_DIR/embeddings
fi

echo "=============================================="
echo "Molecular GNF flow (N-conditioned, early stopping): training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    --train_data_dir molecular_generation/data \
    --val_data_dir $VAL_DIR \
    --logdir $FLOW_LOGDIR \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --num_train_iters 200000 \
    --train_epochs 70 \
    --train_batch_size 32 \
    --eval_every_n_steps 2000 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.05 \
    --random_seed $SEED \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_n_conditioned_qm9_earlystop"

echo "=============================================="
echo "Training stopped"
echo "End time: $(date)"
echo "=============================================="

echo "=============================================="
echo "Generating 10,000 molecules from the best-validation checkpoint"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
    --num_molecules_to_generate 10000 \
    --flow_checkpoint $FLOW_LOGDIR/best/best \
    --random_seed $SEED \
    --save_results_json molecular_generation/results/generated_molecules_n_conditioned_earlystop_10k.json

echo "=============================================="
echo "ALL STEPS COMPLETED"
echo "End time: $(date)"
echo "=============================================="
