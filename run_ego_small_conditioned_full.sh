#!/bin/bash
#SBATCH --job-name=GNF_ego_conditioned
#SBATCH --output=logs/ego_conditioned_output_%a.log
#SBATCH --error=logs/ego_conditioned_error_%a.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2
#SBATCH --array=1-5

# ============================================================
# Full-scale N-conditioned GNF on ego-small, mirroring matta's
# baseline run.sh (results/ego/seed_$SEED) as closely as possible
# for a fair comparison:
#   - Steps 1, 2, 4, 5 unchanged (encoder, embeddings, generate,
#     MMD -- none of these touch the GNF model or prior, so a
#     conditioned checkpoint works with them exactly as-is).
#   - Step 3 swaps train_grevnet_with_data.py (GNFBlock, flat
#     N(0,I) prior) for train_grevnet_conditioned_with_data.py
#     (NConditionedGNFBlock, conditional prior). --num_layers and
#     --attn_num_heads are dropped (the conditioned architecture
#     doesn't use the attention-based GNN factory at all); the
#     rest of the hyperparameters (dataset, node_embedding_dim,
#     num_coupling_layers, iters, epochs, batch size, lr schedule,
#     seed, checkpoint cadence) match the baseline exactly.
#
# 1-5 seed array, matching the baseline's run.sh convention, for
# a statistically rigorous (mean +/- std across seeds) comparison.
# Each seed still generates the full NUMBER_TO_GENERATE=1024 graphs.
# ============================================================
SEED=$SLURM_ARRAY_TASK_ID
PYTHON=/home/matta/miniconda3/envs/gnf_2026/bin/python
WORKDIR=/home/matta/graph-normalising-flow
RESULTS_DIR=$WORKDIR/results/ego_conditioned/seed_$SEED

# ============================================================
# HYPERPARAMETERS  (ego-small paper settings -- matches matta's
# baseline run.sh exactly, except where noted above)
# ============================================================
DATASET=graph_rnn_ego_small
NODE_EMBEDDING_DIM=14
NUM_LAYERS=3
LATENT_DIM=2048
ATTN_NUM_HEADS=8
NUM_TRAIN_ITERS=100000
GNN_LR=1e-04
GNF_LR=1e-04
LR_TYPE=fixed_decay
LR_FIXED_DECAY_RATE=0.99
LR_FIXED_DECAY_STEPS=1000
NUM_COUPLING_LAYERS=12
TRAIN_EPOCHS=15
NUM_PROCESSING_STEPS=10
TRAIN_BATCH_SIZE=8
NUMBER_TO_GENERATE=1024
WANDB_PROJECT=graph-normalising-flow

# Conditioning-specific (no baseline equivalent).
N_EMBED_DIM=32
WEIGHT_SHARING=False
USE_BATCH_NORM=True
PRIOR_KL_WEIGHT=0.1

# ============================================================
# CREATE DIRECTORIES
# ============================================================
mkdir -p $RESULTS_DIR/gnn
mkdir -p $RESULTS_DIR/node_embeddings
mkdir -p $RESULTS_DIR/grevnet_conditioned
mkdir -p $RESULTS_DIR/generated_graphs
mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "DATASET: $DATASET (N-CONDITIONED)"
echo "SEED: $SEED"
echo "RESULTS_DIR: $RESULTS_DIR"
echo "=============================================="

# ============================================================
# STEP 1: Train GNN (unchanged -- conditioning only affects the
# GNF stage, not the encoder).
# ============================================================
echo "=============================================="
echo "STEP 1 STARTED: GNN Training (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u $WORKDIR/run_gnn.py \
    --dataset $DATASET \
    --logdir $RESULTS_DIR/gnn \
    --attn_type dm_attn \
    --num_train_iters $NUM_TRAIN_ITERS \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --latent_dim $LATENT_DIM \
    --num_layers $NUM_LAYERS \
    --attn_num_heads $ATTN_NUM_HEADS \
    --num_processing_steps $NUM_PROCESSING_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --lr $GNN_LR \
    --lr_type $LR_TYPE \
    --lr_fixed_decay_rate $LR_FIXED_DECAY_RATE \
    --lr_fixed_decay_steps $LR_FIXED_DECAY_STEPS \
    --random_seed $SEED \
    --save_every_n_iter 1000 \
    --max_checkpoints_to_keep 5 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "ego_conditioned_seed${SEED}_gnn"

echo "=============================================="
echo "STEP 1 COMPLETED: GNN Training (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 2: Generate Node Embeddings (unchanged).
# ============================================================
echo "=============================================="
echo "STEP 2 STARTED: Generating node embeddings (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

CKPT=$(ls $RESULTS_DIR/gnn/checkpoints-*.index 2>/dev/null | sort -V | tail -1 | sed 's/\.index$//')
echo "Using checkpoint: $CKPT"

srun $PYTHON -u $WORKDIR/generate_grevnet_training_data.py \
    --dataset $DATASET \
    --checkpoint $CKPT \
    --output_file $RESULTS_DIR/node_embeddings/embeddings \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_examples 50000 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --random_seed $SEED \
    --run_number $SEED

echo "=============================================="
echo "STEP 2 COMPLETED: Node embeddings (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 3: Train the N-conditioned GNF (NConditionedGNFBlock +
# conditional prior). This is the only step that differs from
# the baseline script.
# ============================================================
echo "=============================================="
echo "STEP 3 STARTED: N-conditioned GNF Training (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u $WORKDIR/train_grevnet_conditioned_with_data.py \
    --dataset $DATASET \
    --train_data_dir $RESULTS_DIR/node_embeddings \
    --logdir $RESULTS_DIR/grevnet_conditioned \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_coupling_layers $NUM_COUPLING_LAYERS \
    --latent_dim $LATENT_DIM \
    --n_embed_dim $N_EMBED_DIM \
    --weight_sharing $WEIGHT_SHARING \
    --use_batch_norm $USE_BATCH_NORM \
    --prior_kl_weight $PRIOR_KL_WEIGHT \
    --num_train_iters $NUM_TRAIN_ITERS \
    --train_epochs $TRAIN_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --lr $GNF_LR \
    --lr_type $LR_TYPE \
    --lr_fixed_decay_rate $LR_FIXED_DECAY_RATE \
    --lr_fixed_decay_steps $LR_FIXED_DECAY_STEPS \
    --clip_gradient_by_norm False \
    --random_seed $SEED \
    --save_every_n_steps 500 \
    --max_checkpoints_to_keep 5 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "ego_conditioned_seed${SEED}_grevnet"

echo "=============================================="
echo "STEP 3 COMPLETED: GNF Training (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 4: Generate Graphs (unchanged -- restores whatever's in
# the checkpoint's collections generically, works as-is).
# ============================================================
echo "=============================================="
echo "STEP 4 STARTED: Generate graphs (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u $WORKDIR/generate_graphs.py \
    --dataset $DATASET \
    --ckpt_dir $RESULTS_DIR/grevnet_conditioned \
    --output_dir $RESULTS_DIR/generated_graphs \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --number_to_generate $NUMBER_TO_GENERATE \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "ego_conditioned_seed${SEED}_generate"

echo "=============================================="
echo "STEP 4 COMPLETED: Generate graphs (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 5: Compute MMD (unchanged).
# ============================================================
echo "=============================================="
echo "STEP 5 STARTED: MMD (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u $WORKDIR/compute_mmd.py \
    --generated_graphs $RESULTS_DIR/generated_graphs/graphs.p \
    --dataset $DATASET \
    --output_file $RESULTS_DIR/mmd_results.json \
    --graphrnn_eval_dir $WORKDIR/GraphRNN/eval \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "ego_conditioned_seed${SEED}_mmd"

echo "=============================================="
echo "STEP 5 COMPLETED: MMD (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

echo "=============================================="
echo "ALL STEPS COMPLETED for seed $SEED"
echo "End time: $(date)"
echo "=============================================="
