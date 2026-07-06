#!/bin/bash
#SBATCH --job-name=GNF_conditioned_smoke_test
#SBATCH --output=logs/conditioned_smoke_output.log
#SBATCH --error=logs/conditioned_smoke_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Smoke test only: 3 iterations per stage, just to confirm the
# N-conditioned + conditional-prior architecture wires together
# correctly end to end on a real dataset (ego_small) -- not to
# produce a useful trained model.
# ============================================================
SEED=1
PYTHON=/home/matta/miniconda3/envs/gnf_2026/bin/python
WORKDIR=/home/matta/graph-normalising-flow
RESULTS_DIR=$WORKDIR/results/ego_small_conditioned_smoke_test
DATASET=graph_rnn_ego_small
NODE_EMBEDDING_DIM=14
SMOKE_ITERS=3
TRAIN_BATCH_SIZE=8

mkdir -p $RESULTS_DIR/gnn
mkdir -p $RESULTS_DIR/node_embeddings
mkdir -p $RESULTS_DIR/grevnet_conditioned
mkdir -p $RESULTS_DIR/generated_graphs
mkdir -p $WORKDIR/logs

cd $WORKDIR
export WANDB_MODE=offline

echo "=============================================="
echo "DATASET: $DATASET   SEED: $SEED   RESULTS_DIR: $RESULTS_DIR"
echo "=============================================="

# ============================================================
# STAGE 1: Train GNN autoencoder (unchanged -- conditioning only
# affects the GNF stage, not the encoder).
# ============================================================
echo "STAGE 1 STARTED: GNN autoencoder -- $(date)"
STAGE_START=$(date +%s)

srun $PYTHON -u $WORKDIR/run_gnn.py \
    --dataset $DATASET \
    --logdir $RESULTS_DIR/gnn \
    --attn_type dm_attn \
    --num_train_iters $SMOKE_ITERS \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --latent_dim 2048 \
    --num_layers 3 \
    --attn_num_heads 8 \
    --num_processing_steps 10 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --log_every_n_steps 1 \
    --save_every_n_iter 1 \
    --max_checkpoints_to_keep 5 \
    --random_seed $SEED

STAGE_END=$(date +%s)
echo "STAGE 1 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

# ============================================================
# STAGE 2: Generate node embeddings from the (barely trained)
# autoencoder checkpoint.
# ============================================================
echo "STAGE 2 STARTED: node embeddings -- $(date)"
STAGE_START=$(date +%s)

CKPT=$(ls $RESULTS_DIR/gnn/checkpoints-*.index 2>/dev/null | sort -V | tail -1 | sed 's/\.index$//')
echo "Using GNN checkpoint: $CKPT"

srun $PYTHON -u $WORKDIR/generate_grevnet_training_data.py \
    --dataset $DATASET \
    --checkpoint $CKPT \
    --output_file $RESULTS_DIR/node_embeddings/embeddings \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_examples 64 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --random_seed $SEED \
    --run_number $SEED

STAGE_END=$(date +%s)
echo "STAGE 2 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

# ============================================================
# STAGE 3: Train the N-conditioned GNF (NConditionedGNFBlock +
# conditional prior) on those embeddings.
# Note: --sample_size is left at its default (32), NOT
# TRAIN_BATCH_SIZE -- generate_graphs.py in stage 4 hard-codes a
# batch size of 32 when restoring 'sample_n_node_placeholder'
# from the checkpoint, so this must match or stage 4 will fail
# with a shape mismatch.
#
# latent_dim/num_coupling_layers deliberately small here:
# NConditionedGNFBlock has no batch-norm/renormalization step
# between coupling layers (unlike GNFBlock), so at large hidden
# dims + many stacked layers the unclamped affine scale exp(s)
# can overflow across layers and produce NaN losses even at
# iteration 0. Keeping these small avoids that for this smoke
# test; it does not fix the underlying instability at full scale.
# ============================================================
echo "STAGE 3 STARTED: N-conditioned GNF -- $(date)"
STAGE_START=$(date +%s)

srun $PYTHON -u $WORKDIR/train_grevnet_conditioned_with_data.py \
    --dataset $DATASET \
    --train_data_dir $RESULTS_DIR/node_embeddings \
    --logdir $RESULTS_DIR/grevnet_conditioned \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_coupling_layers 4 \
    --latent_dim 64 \
    --n_embed_dim 32 \
    --weight_sharing True \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --train_epochs 1 \
    --num_train_iters $SMOKE_ITERS \
    --save_every_n_steps 1 \
    --log_every_n_steps 1 \
    --random_seed $SEED

STAGE_END=$(date +%s)
echo "STAGE 3 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

# ============================================================
# STAGE 4: Generate graphs from the (barely trained) conditioned
# GNF checkpoint and compute MMD against the test set. Unchanged
# script -- it restores 'sample_pred_adj' / 'sample_log_prob' /
# 'sample_n_node' generically, regardless of which prior produced
# them, so it works with the conditioned checkpoint as-is.
# ============================================================
echo "STAGE 4 STARTED: generate graphs -- $(date)"
STAGE_START=$(date +%s)

srun $PYTHON -u $WORKDIR/generate_graphs.py \
    --dataset $DATASET \
    --ckpt_dir $RESULTS_DIR/grevnet_conditioned \
    --output_dir $RESULTS_DIR/generated_graphs \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --number_to_generate 1

STAGE_END=$(date +%s)
echo "STAGE 4 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

echo "=============================================="
echo "SMOKE TEST DONE for seed $SEED"
echo "=============================================="
