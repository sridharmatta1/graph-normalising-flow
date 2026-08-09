#!/bin/bash
#SBATCH --job-name=GNF_ego_baseline
#SBATCH --output=logs/ego_baseline_output.log
#SBATCH --error=logs/ego_baseline_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

# ============================================================
# Baseline (unconditioned) GNF on ego-small -- single seed, not
# the original run.sh's 1-5 array. One full run already produces
# the target 1024 generated graphs, matching the single-seed
# convention used for the N-conditioned confirmation runs.
# ============================================================
SEED=1
PYTHON=/home/matta/miniconda3/envs/gnf_2026/bin/python
WORKDIR=/home/matta/graph-normalising-flow
RESULTS_DIR=$WORKDIR/results/ego_1024/seed_$SEED

# ============================================================
# HYPERPARAMETERS  (ego-small paper settings -- matches run.sh)
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

# ============================================================
# CREATE DIRECTORIES
# ============================================================
mkdir -p $RESULTS_DIR/gnn
mkdir -p $RESULTS_DIR/node_embeddings
mkdir -p $RESULTS_DIR/grevnet
mkdir -p $RESULTS_DIR/generated_graphs
mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "DATASET: $DATASET"
echo "SEED: $SEED"
echo "RESULTS_DIR: $RESULTS_DIR"
echo "=============================================="

# ============================================================
# STEP 1: Train GNN
# ============================================================
echo "=============================================="
echo "STEP 1 STARTED: GNN Training (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON $WORKDIR/run_gnn.py \
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
    --wandb_run_name "ego_baseline_seed${SEED}_gnn"

echo "=============================================="
echo "STEP 1 COMPLETED: GNN Training (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 2: Generate Node Embeddings
# ============================================================
echo "=============================================="
echo "STEP 2 STARTED: Generating node embeddings (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

CKPT=$(ls $RESULTS_DIR/gnn/checkpoints-*.index 2>/dev/null | sort -V | tail -1 | sed 's/\.index$//')
echo "Using checkpoint: $CKPT"

srun $PYTHON $WORKDIR/generate_grevnet_training_data.py \
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
# STEP 3: Train GRevNet / GNF
# ============================================================
echo "=============================================="
echo "STEP 3 STARTED: GNF Training (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON $WORKDIR/train_grevnet_with_data.py \
    --dataset $DATASET \
    --train_data_dir $RESULTS_DIR/node_embeddings \
    --logdir $RESULTS_DIR/grevnet \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_coupling_layers $NUM_COUPLING_LAYERS \
    --num_layers $NUM_LAYERS \
    --latent_dim $LATENT_DIM \
    --attn_num_heads $ATTN_NUM_HEADS \
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
    --wandb_run_name "ego_baseline_seed${SEED}_grevnet"

echo "=============================================="
echo "STEP 3 COMPLETED: GNF Training (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 4: Generate Graphs
# ============================================================
echo "=============================================="
echo "STEP 4 STARTED: Generate graphs (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON $WORKDIR/generate_graphs.py \
    --dataset $DATASET \
    --ckpt_dir $RESULTS_DIR/grevnet \
    --output_dir $RESULTS_DIR/generated_graphs \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --number_to_generate $NUMBER_TO_GENERATE \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "ego_baseline_seed${SEED}_generate"

echo "=============================================="
echo "STEP 4 COMPLETED: Generate graphs (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

# ============================================================
# STEP 5: Compute MMD
# ============================================================
echo "=============================================="
echo "STEP 5 STARTED: MMD (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON $WORKDIR/compute_mmd.py \
    --generated_graphs $RESULTS_DIR/generated_graphs/graphs.p \
    --dataset $DATASET \
    --output_file $RESULTS_DIR/mmd_results.json \
    --graphrnn_eval_dir $WORKDIR/GraphRNN/eval \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "ego_baseline_seed${SEED}_mmd"

echo "=============================================="
echo "STEP 5 COMPLETED: MMD (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

echo "=============================================="
echo "ALL STEPS COMPLETED for seed $SEED"
echo "End time: $(date)"
echo "=============================================="
